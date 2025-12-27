# Resumo LIME - Centralizado (XGBoost)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Humidity3pm: 0.178792
- Pressure3pm: 0.143022
- Pressure9am: 0.095893
- WindGustSpeed: 0.087434
- MinTemp: 0.043581
- MaxTemp: 0.034094
- Temp3pm: 0.033893
- WindSpeed3pm: 0.032988
- Rainfall: 0.021466
- Humidity9am: 0.017834
- WindDir9am_cos: 0.014868
- Temp9am: 0.013034
- WindDir3pm_cos: 0.012861
- WindSpeed9am: 0.011794
- WindGustDir_cos: 0.009181
- WindDir9am_sin: 0.008230
- WindGustDir_sin: 0.007214
- WindDir3pm_sin: 0.006396
- RainToday: 0.005492

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- WindGustSpeed: classe0=0.086950, classe1=0.089135, diff=-0.002186
- MaxTemp: classe0=0.033725, classe1=0.035391, diff=-0.001666
- Temp3pm: classe0=0.033526, classe1=0.035182, diff=-0.001655
- Humidity3pm: classe0=0.178454, classe1=0.179980, diff=-0.001526
- Rainfall: classe0=0.021176, classe1=0.022487, diff=-0.001312
- Pressure9am: classe0=0.095675, classe1=0.096659, diff=-0.000984
- Temp9am: classe0=0.012842, classe1=0.013711, diff=-0.000869
- WindGustDir_cos: classe0=0.008995, classe1=0.009833, diff=-0.000838
- WindDir9am_cos: classe0=0.014998, classe1=0.014412, diff=0.000586
- Humidity9am: classe0=0.017712, classe1=0.018262, diff=-0.000550

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_xgb_class_0_mean_abs_weight_top19.png
- lime_xgb_class_1_mean_abs_weight_top19.png
- lime_xgb_feature_weight_distribution_top19.png
- lime_xgb_global_mean_abs_weight_top19.png
- lime_xgb_heatmap_top19_rows200.png
- lime_xgb_heatmap_top19_rows300.png
- lime_xgb_waterfall_idx25435_label1.png
- lime_xgb_waterfall_idx31772_label0.png
- lime_xgb_waterfall_idx68270_label1.png
- lime_xgb_waterfall_idx81390_label0.png
- summary.md
