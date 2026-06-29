# Training And Evaluation

`training/` owns generated target corpora, calibration gates, value-net training,
strength measurement, tablebase tooling, and reanalysis flows. It may depend on
`src/`, `environment/`, and `hal/`, but runtime engine code should not import it.
