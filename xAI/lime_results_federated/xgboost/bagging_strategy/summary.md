# Resumo LIME - XGBoost Federado (Bagging)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Humidity3pm: 0.276963
- Pressure3pm: 0.087259
- WindGustSpeed: 0.076209
- Rainfall: 0.023960
- Pressure9am: 0.019119
- MinTemp: 0.016744
- WindGustDir_cos: 0.016279
- Temp3pm: 0.014427
- WindSpeed9am: 0.007278
- WindSpeed3pm: 0.007264
- Temp9am: 0.007084
- WindDir9am_sin: 0.007045
- WindDir3pm_cos: 0.006805
- RainToday: 0.006775
- WindDir9am_cos: 0.006742
- Humidity9am: 0.006732
- MaxTemp: 0.006641
- WindDir3pm_sin: 0.006616
- WindGustDir_sin: 0.006593

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Humidity3pm: classe0=0.275299, classe1=0.282810, diff=-0.007511
- Pressure3pm: classe0=0.086851, classe1=0.088694, diff=-0.001843
- WindGustSpeed: classe0=0.075993, classe1=0.076968, diff=-0.000975
- Temp3pm: classe0=0.014251, classe1=0.015047, diff=-0.000796
- WindGustDir_cos: classe0=0.016146, classe1=0.016748, diff=-0.000602
- WindDir9am_cos: classe0=0.006640, classe1=0.007099, diff=-0.000459
- Pressure9am: classe0=0.019209, classe1=0.018803, diff=0.000406
- Humidity9am: classe0=0.006803, classe1=0.006482, diff=0.000322
- MaxTemp: classe0=0.006692, classe1=0.006463, diff=0.000229
- WindGustDir_sin: classe0=0.006551, classe1=0.006742, diff=-0.000191

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
- lime_xgb_bagging_heatmap_top19_rows300.png
- lime_xgb_bagging_waterfall_idx74993_label0.png
- lime_xgb_bagging_waterfall_idx87299_label1.png
