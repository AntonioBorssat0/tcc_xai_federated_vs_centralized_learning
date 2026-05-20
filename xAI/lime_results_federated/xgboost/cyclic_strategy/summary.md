# Resumo LIME - XGBoost Federado (Cyclic)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.140968
- Humidity3pm: 0.139606
- Temp3pm: 0.098151
- WindGustSpeed: 0.077661
- Pressure9am: 0.060968
- Humidity9am: 0.042887
- WindDir9am_cos: 0.033837
- Temp9am: 0.033596
- MinTemp: 0.031260
- WindDir3pm_cos: 0.026794
- Rainfall: 0.021981
- WindSpeed3pm: 0.019868
- WindDir3pm_sin: 0.016558
- WindDir9am_sin: 0.010908
- WindSpeed9am: 0.010796
- WindGustDir_cos: 0.009691
- WindGustDir_sin: 0.008418
- MaxTemp: 0.007131
- RainToday: 0.006323

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Humidity3pm: classe0=0.138904, classe1=0.142070, diff=-0.003166
- Pressure9am: classe0=0.060336, classe1=0.063189, diff=-0.002854
- Temp3pm: classe0=0.097768, classe1=0.099498, diff=-0.001729
- WindDir9am_sin: classe0=0.010639, classe1=0.011855, diff=-0.001216
- WindDir3pm_sin: classe0=0.016371, classe1=0.017214, diff=-0.000842
- WindDir9am_cos: classe0=0.034008, classe1=0.033234, diff=0.000774
- WindGustSpeed: classe0=0.077491, classe1=0.078256, diff=-0.000765
- WindDir3pm_cos: classe0=0.026961, classe1=0.026209, diff=0.000752
- Rainfall: classe0=0.022129, classe1=0.021463, diff=0.000665
- Pressure3pm: classe0=0.141063, classe1=0.140633, diff=0.000430

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_xgb_cyclic_class_0_mean_abs_weight_top19.png
- lime_xgb_cyclic_class_1_mean_abs_weight_top19.png
- lime_xgb_cyclic_feature_weight_distribution_top19.png
- lime_xgb_cyclic_global_mean_abs_weight_top19.png
- lime_xgb_cyclic_heatmap_top19_rows300.png
- lime_xgb_cyclic_waterfall_idx44546_label0.png
- lime_xgb_cyclic_waterfall_idx62358_label1.png
