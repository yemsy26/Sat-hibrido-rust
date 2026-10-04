import { Divider, H1, H2, Stack, Table, Text } from 'cursor/canvas';

export default function PDOBenchmarkComparison() {
  return (
    <Stack gap={16} style={{ maxWidth: 1060, margin: '0 auto', padding: 24 }}>
      <H1>Fase 4 frente a Fase 5 · PDO</H1>
      <Text>cargo run --release: compilación y ejecución correctas. Ambas pruebas del palomar devolvieron UNSAT.</Text>
      <H2>Comparación con los registros originales</H2>
      <Table
        headers={['Prueba', 'Métrica', 'Antes · Fase 4', 'Ahora · Fase 5 PDO', 'Reducción']}
        columnAlign={['left', 'left', 'right', 'right', 'right']}
        rows={[
          ['pigeonhole_6_5', 'elapsed_ms', '1.462700', '0.380800', '73,97 %'],
          ['pigeonhole_6_5', 'decisions', '197', '180', '8,63 %'],
          ['pigeonhole_6_5', 'conflicts', '182', '174', '4,40 %'],
          ['pigeonhole_7_6', 'elapsed_ms', '12.014100', '3.417900', '71,55 %'],
          ['pigeonhole_7_6', 'decisions', '1262', '1199', '4,99 %'],
          ['pigeonhole_7_6', 'conflicts', '1148', '1044', '9,06 %'],
        ]}
      />
      <Text tone="secondary" size="small">Antes: registros de la ejecución inicial de Fase 4. Ahora: CSV capturado en la ejecución actual. elapsed_ms mide tiempo de pared, no ciclos de CPU.</Text>
      <Divider />
      <H2>Control temporal: Fase 4 ejecutada nuevamente</H2>
      <Table
        headers={['Prueba', 'Fase 4 · control actual (ms)', 'PDO · actual (ms)', 'Reducción']}
        columnAlign={['left', 'right', 'right', 'right']}
        rows={[
          ['pigeonhole_6_5', '0.508900', '0.380800', '25,17 %'],
          ['pigeonhole_7_6', '4.744300', '3.417900', '27,96 %'],
        ]}
      />
      <Text tone="secondary" size="small">Una ejecución por versión; el control conserva los conteos originales. La diferencia entre tiempos históricos y actuales muestra variación de las condiciones de ejecución. Estas mediciones no aíslan causalmente todos los efectos de PDO.</Text>
      <H2>Conclusión</H2>
      <Text>PDO redujo decisiones y conflictos en ambas instancias: menos trabajo de búsqueda observado. Esto no demuestra una reducción matemática general del árbol ni una mejora asintótica.</Text>
    </Stack>
  );
}
