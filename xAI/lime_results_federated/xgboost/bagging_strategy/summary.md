# Resumo LIME - XGBoost Federado (Bagging)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Humidity3pm: 0.212853
- Pressure3pm: 0.059376
- WindGustSpeed: 0.051776
- Rainfall: 0.025360
- WindDir3pm_cos: 0.020845
- Temp3pm: 0.020170
- Pressure9am: 0.019726
- MinTemp: 0.012073
- Humidity9am: 0.007443
- WindGustDir_cos: 0.005870
- WindSpeed3pm: 0.005776
- WindSpeed9am: 0.005594
- WindDir9am_sin: 0.005563
- WindGustDir_sin: 0.005326
- RainToday: 0.005313
- Temp9am: 0.005296
- WindDir3pm_sin: 0.005286
- WindDir9am_cos: 0.005285
- MaxTemp: 0.005271

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Humidity3pm: classe0=0.210346, classe1=0.221668, diff=-0.011323
- WindGustSpeed: classe0=0.051394, classe1=0.053121, diff=-0.001727
- Temp3pm: classe0=0.019791, classe1=0.021503, diff=-0.001712
- Pressure3pm: classe0=0.059015, classe1=0.060646, diff=-0.001631
- Pressure9am: classe0=0.019551, classe1=0.020341, diff=-0.000790
- Humidity9am: classe0=0.007313, classe1=0.007903, diff=-0.000591
- WindDir9am_cos: classe0=0.005192, classe1=0.005610, diff=-0.000417
- WindGustDir_sin: classe0=0.005244, classe1=0.005615, diff=-0.000371
- WindSpeed9am: classe0=0.005524, classe1=0.005839, diff=-0.000315
- WindDir3pm_cos: classe0=0.020782, classe1=0.021064, diff=-0.000282

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_xgb_bagging_class_0_mean_abs_weight_top19.png
- lime_xgb_bagging_class_1_mean_abs_weight_top19.png
- lime_xgb_bagging_feature_weight_distribution_top19.png
- lime_xgb_bagging_global_mean_abs_weight_top19.png
- lime_xgb_bagging_heatmap_top19_rows200.png
- lime_xgb_bagging_heatmap_top19_rows300.png
- lime_xgb_bagging_waterfall_idx100001_label0.png
- lime_xgb_bagging_waterfall_idx101738_label1.png
- lime_xgb_bagging_waterfall_idx12999_label1.png
- lime_xgb_bagging_waterfall_idx40595_label1.png
- lime_xgb_bagging_waterfall_idx41030_label0.png
- lime_xgb_bagging_waterfall_idx42430_label1.png
- lime_xgb_bagging_waterfall_idx54550_label0.png
- lime_xgb_bagging_waterfall_idx66633_label0.png
- summary.md
