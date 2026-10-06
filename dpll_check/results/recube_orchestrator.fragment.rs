// Optional dynamic frontier. The GPU callback stays on the calling thread.
struct RecubeJob {
    cube: Vec<u32>,
    conflict_limit: usize,
    started: bool,
}

enum RecubeEvent {
    Finished {
        owner: usize,
        job: RecubeJob,
        first_start: bool,
        outcome: SearchOutcome,
        total: SolveStats,
        split: Vec<u32>,
    },
    Failed { owner: usize },
    Stopped { owner: usize },
}

fn recube_worker(
    base: &State,
    variables: &[u32],
    config: SwarmConfig,
    recube: &RecubeConfig,
    owner: usize,
    cancel: &AtomicBool,
    pool: &CloudPool,
    progress: Option<&SwarmProgress>,
    jobs: &std::sync::mpsc::Receiver<RecubeJob>,
    events: &std::sync::mpsc::Sender<RecubeEvent>,
) {
    let mut state = base.clone();
    if let Some(walker) = state.walker.as_mut() {
        walker.reseed(owner as u64 + 1);
    }
    let feedback = active(Some(config.reignition));
    let mut link = (config.share_enabled && config.share_capacity != 0)
        .then(|| Sharing::new(pool, owner as u32, config.share_interval.max(1) as u64));
    while !cancel.load(Ordering::Acquire) {
        let Ok(mut job) = jobs.recv() else { break; };
        if cancel.load(Ordering::Acquire) { break; }
        if !state.root_unsat {
            debug_assert_eq!(state.head, state.trail.len());
            state.backjump(0);
        }
        let first_start = !job.started;
        job.started = true;
        state.stats = SolveStats::default();
        let control = SearchControl {
            cube: &job.cube, cancel: Some(cancel), sharing: link.as_mut(), proof: None,
            progress: progress.map(Beacon::new),
        };
        let (mut outcome, total) = run_state_budget::<true>(
            &mut state, variables, feedback, control, job.conflict_limit,
        );
        let mut split = Vec::new();
        match &outcome {
            SearchOutcome::Yielded => {
                // Yielded searches have returned to the root propagation fixpoint.
                debug_assert_eq!(state.level(), 0);
                debug_assert_eq!(state.head, state.trail.len());
                let mut candidates: Vec<usize> = (0..variables.len()).filter(|&variable| {
                    state.values[variable << 1] == 0
                        && variables[variable] <= recube.max_split_variable
                        && !job.cube.iter().any(|&literal| (literal >> 1) as usize == variable)
                }).collect();
                candidates.sort_unstable_by(|&a, &b| {
                    state.scores[b].total_cmp(&state.scores[a]).then_with(|| a.cmp(&b))
                });
                candidates.truncate(recube.split_variables.min(16));
                split.extend(candidates.into_iter().map(|variable| {
                    ((variable as u32) << 1) | u32::from(state.phases[variable])
                }));
            }
            SearchOutcome::Sat(_) => {
                if cancel.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
                    outcome = SearchOutcome::Cancelled;
                }
            }
            SearchOutcome::Unsat => {
                if let Some(progress) = progress {
                    progress.cubes_closed.fetch_add(1, Ordering::Relaxed);
                }
            }
            SearchOutcome::Cancelled => {}
        }
        if events.send(RecubeEvent::Finished {
            owner, job, first_start, outcome, total, split,
        }).is_err() { break; }
    }
}

// A malformed callback response closes no cube, matching the static path.
fn recube_filter_jobs(
    jobs: Vec<RecubeJob>,
    variables: &[u32],
    filter: &mut Option<CubeFilter<'_>>,
    stats: &mut SwarmStats,
    progress: Option<&SwarmProgress>,
) -> (Vec<RecubeJob>, Option<Vec<i32>>, usize) {
    let Some(callback) = filter.as_mut() else { return (jobs, None, 0); };
    let mut line = Vec::new();
    let asked: Vec<Vec<i32>> = jobs.iter().map(|job| {
        external(variables, &job.cube, &mut line);
        line.clone()
    }).collect();
    let verdicts = (**callback)(&asked);
    if verdicts.len() != jobs.len() { return (jobs, None, 0); }
    let mut open = Vec::with_capacity(jobs.len());
    let mut model = None;
    let mut closed = 0;
    for (job, verdict) in jobs.into_iter().zip(verdicts) {
        match verdict {
            CubeVerdict::Unknown => open.push(job),
            CubeVerdict::Unsat => closed += 1,
            CubeVerdict::Sat(found) => {
                if model.is_none() { model = Some(found); }
            }
        }
    }
    stats.cubes_filtered += closed;
    if let Some(progress) = progress {
        progress.cubes_filtered.store(stats.cubes_filtered as u64, Ordering::Relaxed);
    }
    (open, model, closed)
}

struct RecubeFrontier {
    ready: VecDeque<RecubeJob>,
    filter_pending: VecDeque<RecubeJob>,
    outstanding: usize,
    limit: usize,
    busy: Vec<bool>,
    stopped: Vec<bool>,
    winner: Option<Vec<i32>>,
    failed_owner: Option<usize>,
}

// Unwinding a GPU callback must cancel busy workers before scope joins them.
struct RecubeCancelOnDrop<'a>(&'a AtomicBool);
impl Drop for RecubeCancelOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl RecubeFrontier {
    fn dispatch(
        &mut self,
        owner: usize,
        senders: &[std::sync::mpsc::SyncSender<RecubeJob>],
        cancel: &AtomicBool,
    ) {
        if cancel.load(Ordering::Acquire) || self.busy[owner] || self.stopped[owner] { return; }
        let Some(sender) = senders.get(owner) else { return; };
        let Some(job) = self.ready.pop_front() else { return; };
        match sender.try_send(job) {
            Ok(()) => self.busy[owner] = true,
            Err(std::sync::mpsc::TrySendError::Full(job)) => self.ready.push_front(job),
            Err(std::sync::mpsc::TrySendError::Disconnected(job)) => {
                self.ready.push_front(job);
                self.failed_owner = Some(owner);
                cancel.store(true, Ordering::Release);
            }
        }
    }

    fn record(
        &mut self,
        event: RecubeEvent,
        stats: &mut SwarmStats,
        progress: Option<&SwarmProgress>,
        recube: &RecubeConfig,
        cancel: &AtomicBool,
        senders: &[std::sync::mpsc::SyncSender<RecubeJob>],
    ) {
        match event {
            RecubeEvent::Failed { owner } => {
                self.failed_owner = Some(owner);
                cancel.store(true, Ordering::Release);
            }
            RecubeEvent::Stopped { owner } => {
                assert!(!self.stopped[owner], "Duplicate recubing worker exit");
                self.stopped[owner] = true;
                if !senders.is_empty() && !cancel.load(Ordering::Acquire) {
                    self.failed_owner = Some(owner);
                    cancel.store(true, Ordering::Release);
                }
            }
            RecubeEvent::Finished { owner, mut job, first_start, outcome, total, split } => {
                assert!(self.busy[owner], "A recubing report must own an in-flight job");
                self.busy[owner] = false;
                stats.cubes_started += usize::from(first_start);
                stats.total.add_search(total);
                match outcome {
                    SearchOutcome::Unsat => {
                        stats.cubes_unsat += 1;
                        self.outstanding = self.outstanding.checked_sub(1).expect("Live cube underflow");
                    }
                    SearchOutcome::Sat(model) => {
                        self.winner = Some(model);
                        cancel.store(true, Ordering::Release);
                    }
                    SearchOutcome::Cancelled => stats.cubes_cancelled += 1,
                    SearchOutcome::Yielded => {
                        stats.cubes_yielded += 1;
                        if let Some(progress) = progress {
                            progress.cubes_yielded.fetch_add(1, Ordering::Relaxed);
                        }
                        if !cancel.load(Ordering::Acquire) {
                            let count = 1usize << split.len();
                            let expanded = self.outstanding.checked_add(count - 1);
                            if split.is_empty() || expanded.is_none_or(|live| live > self.limit) {
                                // Keep the leaf unresolved and remove its next search budget.
                                job.conflict_limit = 0;
                                self.ready.push_front(job);
                            } else {
                                self.outstanding = expanded.expect("Checked live cube count");
                                stats.cubes_generated = stats.cubes_generated.checked_add(count)
                                    .expect("Generated cube count overflow");
                                stats.cubes_split += 1;
                                if let Some(progress) = progress {
                                    progress.cubes_split.fetch_add(1, Ordering::Relaxed);
                                    progress.cubes_total.store(
                                        (stats.cubes_generated - stats.cubes_split) as u64,
                                        Ordering::Relaxed,
                                    );
                                }
                                // Replace the parent with its entire disjoint partition.
                                for mask in 0..count {
                                    let mut cube = job.cube.clone();
                                    cube.extend(split.iter().enumerate().map(|(bit, &literal)| {
                                        literal ^ ((mask >> bit) & 1) as u32
                                    }));
                                    self.filter_pending.push_back(RecubeJob {
                                        cube, conflict_limit: recube.max_conflicts_per_cube, started: false,
                                    });
                                }
                            }
                        }
                    }
                }
                // Give this worker existing CPU work before a synchronous GPU call.
                self.dispatch(owner, senders, cancel);
            }
        }
    }
}

pub fn solve_swarm_with_recubing(
    formula: &[Vec<i32>],
    config: SwarmConfig,
    recube: RecubeConfig,
    progress: Option<&SwarmProgress>,
    mut filter: Option<CubeFilter<'_>>,
) -> (Option<Vec<i32>>, SwarmStats) {
    if recube.max_conflicts_per_cube == 0 {
        return solve_swarm_with_filter(formula, config, progress, filter);
    }
    let mut stats = SwarmStats::default();
    let feedback = active(Some(config.reignition));
    let Some((mut base, variables)) = prepare_formula(
        formula, feedback.map(|configuration| configuration.oracle),
    ) else {
        stats.total.conflicts = 1;
        return (None, stats);
    };
    if !base.root_unsat && base.propagate().is_some() {
        base.root_unsat = true;
        base.stats.conflicts += 1;
    }
    if base.root_unsat {
        stats.total = base.stats;
        return (None, stats);
    }
    if base.trail.len() == variables.len() {
        stats.total = base.stats;
        return (Some(extract_model(&base, &variables)), stats);
    }
    if let Some(configuration) = feedback {
        if base.reignite(configuration, None) {
            stats.total = base.stats;
            return (Some(base.oracle_model(&variables)), stats);
        }
    }
    stats.total = base.stats;
    let jobs: Vec<RecubeJob> = make_cubes(&base, config).into_iter().map(|cube| RecubeJob {
        cube, conflict_limit: recube.max_conflicts_per_cube, started: false,
    }).collect();
    stats.cubes_generated = jobs.len();
    if let Some(progress) = progress {
        progress.cubes_total.store(jobs.len() as u64, Ordering::Relaxed);
    }
    let (jobs, model, _) = recube_filter_jobs(jobs, &variables, &mut filter, &mut stats, progress);
    if model.is_some() { return (model, stats); }
    if jobs.is_empty() {
        assert_eq!(stats.cubes_filtered, stats.cubes_generated);
        return (None, stats);
    }
    let workers = config.workers.max(1).min(jobs.len());
    let cancel = AtomicBool::new(false);
    let pool = CloudPool::new(config.share_capacity);
    let winner = std::thread::scope(|scope| {
        let _cancel_on_exit = RecubeCancelOnDrop(&cancel);
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let mut senders = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for owner in 0..workers {
            let (jobs_tx, jobs_rx) = std::sync::mpsc::sync_channel(1);
            let events_tx = events_tx.clone();
            let (base, variables, recube, cancel, pool) =
                (&base, variables.as_slice(), &recube, &cancel, &pool);
            let worker = move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    recube_worker(base, variables, config, recube, owner, cancel, pool,
                        progress, &jobs_rx, &events_tx);
                }));
                if result.is_err() {
                    cancel.store(true, Ordering::Release);
                    let _ = events_tx.send(RecubeEvent::Failed { owner });
                }
                let _ = events_tx.send(RecubeEvent::Stopped { owner });
            };
            match std::thread::Builder::new().name(format!("recube-{owner}"))
                .spawn_scoped(scope, worker) {
                Ok(handle) => { senders.push(jobs_tx); handles.push(handle); }
                Err(_) => break,
            }
        }
        drop(events_tx);
        assert!(!handles.is_empty(), "Unable to start a recubing worker");
        stats.workers = handles.len();
        let mut frontier = RecubeFrontier {
            outstanding: jobs.len(), limit: recube.max_live_cubes.max(jobs.len()),
            ready: jobs.into(), filter_pending: VecDeque::new(),
            busy: vec![false; handles.len()], stopped: vec![false; handles.len()],
            winner: None, failed_owner: None,
        };
        loop {
            while let Ok(event) = events_rx.try_recv() {
                frontier.record(event, &mut stats, progress, &recube, &cancel, &senders);
            }
            if cancel.load(Ordering::Acquire) {
                senders.clear();
                if frontier.stopped.iter().all(|&stopped| stopped) { break; }
            } else {
                if frontier.outstanding == 0 {
                    assert!(frontier.ready.is_empty() && frontier.filter_pending.is_empty());
                    assert!(frontier.busy.iter().all(|&busy| !busy));
                    break;
                }
                for owner in 0..senders.len() { frontier.dispatch(owner, &senders, &cancel); }
                if !cancel.load(Ordering::Acquire) && !frontier.filter_pending.is_empty() {
                    let count = frontier.filter_pending.len().min(1024);
                    let batch = frontier.filter_pending.drain(..count).collect();
                    let (open, model, closed) = recube_filter_jobs(
                        batch, &variables, &mut filter, &mut stats, progress,
                    );
                    frontier.outstanding = frontier.outstanding.checked_sub(closed)
                        .expect("GPU closed more leaves than exist");
                    frontier.ready.extend(open);
                    if let Some(model) = model {
                        frontier.winner = Some(model);
                        cancel.store(true, Ordering::Release);
                    }
                    continue;
                }
            }
            match events_rx.recv() {
                Ok(event) => frontier.record(event, &mut stats, progress, &recube, &cancel, &senders),
                Err(_) => {
                    frontier.failed_owner.get_or_insert(usize::MAX);
                    cancel.store(true, Ordering::Release);
                    break;
                }
            }
        }
        senders.clear();
        for (owner, handle) in handles.into_iter().enumerate() {
            if handle.join().is_err() { frontier.failed_owner = Some(owner); }
        }
        // Reports arriving after SAT still contribute each attempt's final statistics.
        for event in events_rx.try_iter() {
            frontier.record(event, &mut stats, progress, &recube, &cancel, &senders);
        }
        assert!(frontier.failed_owner.is_none(), "A recubing worker failed");
        if frontier.winner.is_none() {
            assert!(!cancel.load(Ordering::Acquire), "Cancelled recubing cannot prove UNSAT");
            assert_eq!(frontier.outstanding, 0, "Unresolved recubing leaves cannot prove UNSAT");
        }
        frontier.winner
    });
    stats.shared_clauses = pool.snapshot(&variables);
    if winner.is_none() {
        assert_eq!(stats.cubes_unsat + stats.cubes_filtered + stats.cubes_split,
            stats.cubes_generated, "Incomplete dynamic cube coverage cannot prove UNSAT");
    }
    (winner, stats)
}
