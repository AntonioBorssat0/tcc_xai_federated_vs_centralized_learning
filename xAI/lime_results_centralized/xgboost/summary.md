# Resumo LIME - Centralizado (XGBoost)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Humidity3pm: 0.183799
- Pressure3pm: 0.155561
- Pressure9am: 0.111115
- WindGustSpeed: 0.094923
- MinTemp: 0.049753
- MaxTemp: 0.046382
- WindSpeed3pm: 0.029753
- Temp3pm: 0.029098
- Humidity9am: 0.020284
- Rainfall: 0.018115
- WindDir9am_cos: 0.017592
- WindDir3pm_cos: 0.014768
- WindSpeed9am: 0.014126
- Temp9am: 0.010653
- WindGustDir_cos: 0.007484
- WindDir3pm_sin: 0.007327
- WindDir9am_sin: 0.006946
- WindGustDir_sin: 0.006500
- RainToday: 0.005619

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Pressure3pm: classe0=0.156023, classe1=0.153939, diff=0.002085
- MaxTemp: classe0=0.046034, classe1=0.047604, diff=-0.001570
- Rainfall: classe0=0.017813, classe1=0.019179, diff=-0.001367
- Humidity3pm: classe0=0.183520, classe1=0.184781, diff=-0.001262
- WindGustSpeed: classe0=0.094679, classe1=0.095780, diff=-0.001100
- Temp3pm: classe0=0.028921, classe1=0.029719, diff=-0.000798
- Temp9am: classe0=0.010539, classe1=0.011052, diff=-0.000513
- WindGustDir_cos: classe0=0.007384, classe1=0.007836, diff=-0.000452
- WindDir3pm_sin: classe0=0.007257, classe1=0.007572, diff=-0.000316
- WindDir9am_cos: classe0=0.017661, classe1=0.017347, diff=0.000315

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
- lime_xgb_heatmap_top19_rows300.png
- lime_xgb_waterfall_idx41030_label0.png
- lime_xgb_waterfall_idx41784_label1.png
