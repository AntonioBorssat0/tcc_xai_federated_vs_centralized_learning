
# Script de Sincronização de Hiperparâmetros
# Sincroniza melhores hiperparâmetros do treino centralizado para configs federados


$projectRoot = $PSScriptRoot

# Funções auxiliares de log
function Write-Header { param([string]$msg) Write-Host $msg -ForegroundColor Cyan }
function Write-Info { param([string]$msg) Write-Host "  [INFO] $msg" -ForegroundColor Gray }
function Write-Success { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }

# FUNÇÃO: Atualizar arquivo TOML com par chave-valor

function Update-TomlValue {
    param(
        [string]$FilePath,
        [string]$Key,
        [string]$Value,
        [string]$Section = ""
    )
    
    $content = Get-Content $FilePath -Raw
    
    # Escape special regex characters in key
    $escapedKey = [regex]::Escape($Key)
    
    # Pattern to match the key with current value
    $patternWithValue = "(?m)^(\s*)$escapedKey\s*=\s*(.+)$"
    
    # Check if key exists and get current value
    if ($content -match $patternWithValue) {
        $currentValue = $Matches[2].Trim()
        
        # Compare values (handle numeric precision)
        $valueToCompare = $Value.ToString().Trim()
        if ($currentValue -eq $valueToCompare) {
            # Value already correct - this is success!
            return @{Success = $true; AlreadyCorrect = $true}
        }
        
        # Value needs update
        $replacement = "`${1}$Key = $Value"
        $newContent = $content -replace $patternWithValue, $replacement
        
        if (-not $DryRun) {
            $newContent | Set-Content $FilePath -NoNewline
        }
        
        return @{Success = $true; AlreadyCorrect = $false}
    } else {
        # Key not found in file
        Write-Warning "Key $Key not found in $FilePath"
        return @{Success = $false; AlreadyCorrect = $false}
    }
}


# SINCRONIZAR HIPERPARÂMETROS MLP

Write-Header "`n[1/2] Hiperparâmetros MLP"

$mlpJsonPath = "$projectRoot\centralized_training\models\mlp\best_hyperparameters.json"
$mlpTomlPath = "$projectRoot\flwr-mlp\pyproject.toml"

if (-not (Test-Path $mlpJsonPath)) {
    Write-Error "MLP hyperparameters not found: $mlpJsonPath"
    Write-Info "Run centralized MLP training first: poetry run python centralized_training\train_centralized_mlp.py"
    $mlpSuccess = $false
} else {
    Write-Success "Found: $mlpJsonPath"
    
    # Read JSON
    $mlpParams = Get-Content $mlpJsonPath | ConvertFrom-Json
    
    if ($Verbose) {
        Write-Info "Loaded parameters:"
        Write-Host "    hidden1: $($mlpParams.hidden1)" -ForegroundColor Gray
        Write-Host "    hidden2: $($mlpParams.hidden2)" -ForegroundColor Gray
        Write-Host "    hidden3: $($mlpParams.hidden3)" -ForegroundColor Gray
        Write-Host "    dropout1: $($mlpParams.dropout1)" -ForegroundColor Gray
        Write-Host "    dropout2: $($mlpParams.dropout2)" -ForegroundColor Gray
        Write-Host "    lr: $($mlpParams.lr)" -ForegroundColor Gray
        Write-Host "    batch_size: $($mlpParams.batch_size)" -ForegroundColor Gray
    }
    
    # Update TOML
    Write-Info "Updating: $mlpTomlPath"
    
    $updates = @(
        @{Key = "hidden-layer-1"; Value = $mlpParams.hidden1},
        @{Key = "hidden-layer-2"; Value = $mlpParams.hidden2},
        @{Key = "hidden-layer-3"; Value = $mlpParams.hidden3},
        @{Key = "dropout-1"; Value = $mlpParams.dropout1},
        @{Key = "dropout-2"; Value = $mlpParams.dropout2},
        @{Key = "learning-rate"; Value = $mlpParams.lr},
        @{Key = "batch-size"; Value = $mlpParams.batch_size}
    )
    
    $updateCount = 0
    foreach ($update in $updates) {
        if ($DryRun) {
            Write-Host "    [DRY-RUN] Would update: $($update.Key) = $($update.Value)" -ForegroundColor Yellow
            $updateCount++
        } else {
            $result = Update-TomlValue -FilePath $mlpTomlPath -Key $update.Key -Value $update.Value
            if ($result.Success) {
                if ($result.AlreadyCorrect) {
                    Write-Host "    [OK] $($update.Key) = $($update.Value) (already correct)" -ForegroundColor Gray
                } else {
                    Write-Host "    [OK] $($update.Key) = $($update.Value)" -ForegroundColor Green
                }
                $updateCount++
            }
        }
    }
    
    if ($updateCount -eq $updates.Count) {
        Write-Success "MLP config updated ($updateCount/$($updates.Count) parameters)"
        $mlpSuccess = $true
    } else {
        Write-Warning "MLP config partially updated ($updateCount/$($updates.Count) parameters)"
        $mlpSuccess = $false
    }
}


# SINCRONIZAR HIPERPARÂMETROS XGBOOST

Write-Header "`n[2/2] Hiperparâmetros XGBoost"

$xgbJsonPath = "$projectRoot\centralized_training\models\xgboost\best_hyperparameters.json"
$xgbTomlPath = "$projectRoot\flwr-xgboost\pyproject.toml"

if (-not (Test-Path $xgbJsonPath)) {
    Write-Error "XGBoost hyperparameters not found: $xgbJsonPath"
    Write-Info "Run centralized XGBoost training first: poetry run python centralized_training\train_centralized_xgboost.py"
    $xgbSuccess = $false
} else {
    Write-Success "Found: $xgbJsonPath"
    
    # Read JSON
    $xgbParams = Get-Content $xgbJsonPath | ConvertFrom-Json
    
    if ($Verbose) {
        Write-Info "Loaded parameters:"
        Write-Host "    max_depth: $($xgbParams.max_depth)" -ForegroundColor Gray
        Write-Host "    learning_rate: $($xgbParams.learning_rate)" -ForegroundColor Gray
        Write-Host "    subsample: $($xgbParams.subsample)" -ForegroundColor Gray
        Write-Host "    colsample_bytree: $($xgbParams.colsample_bytree)" -ForegroundColor Gray
        Write-Host "    min_child_weight: $($xgbParams.min_child_weight)" -ForegroundColor Gray
        Write-Host "    gamma: $($xgbParams.gamma)" -ForegroundColor Gray
        Write-Host "    reg_alpha: $($xgbParams.reg_alpha)" -ForegroundColor Gray
        Write-Host "    reg_lambda: $($xgbParams.reg_lambda)" -ForegroundColor Gray
        Write-Host "    scale_pos_weight: $($xgbParams.scale_pos_weight)" -ForegroundColor Gray
    }
    
    # Update TOML
    Write-Info "Updating: $xgbTomlPath"
    
    $updates = @(
        @{Key = "params.eta"; Value = $xgbParams.learning_rate},
        @{Key = "params.max-depth"; Value = $xgbParams.max_depth},
        @{Key = "params.min-child-weight"; Value = $xgbParams.min_child_weight},
        @{Key = "params.gamma"; Value = $xgbParams.gamma},
        @{Key = "params.subsample"; Value = $xgbParams.subsample},
        @{Key = "params.colsample-bytree"; Value = $xgbParams.colsample_bytree},
        @{Key = "params.reg-alpha"; Value = $xgbParams.reg_alpha},
        @{Key = "params.reg-lambda"; Value = $xgbParams.reg_lambda},
        @{Key = "params.scale-pos-weight"; Value = $xgbParams.scale_pos_weight}
    )
    
    $updateCount = 0
    foreach ($update in $updates) {
        if ($DryRun) {
            Write-Host "    [DRY-RUN] Would update: $($update.Key) = $($update.Value)" -ForegroundColor Yellow
            $updateCount++
        } else {
            $result = Update-TomlValue -FilePath $xgbTomlPath -Key $update.Key -Value $update.Value
            if ($result.Success) {
                if ($result.AlreadyCorrect) {
                    Write-Host "    [OK] $($update.Key) = $($update.Value) (already correct)" -ForegroundColor Gray
                } else {
                    Write-Host "    [OK] $($update.Key) = $($update.Value)" -ForegroundColor Green
                }
                $updateCount++
            }
        }
    }
    
    if ($updateCount -eq $updates.Count) {
        Write-Success "XGBoost config updated ($updateCount/$($updates.Count) parameters)"
        $xgbSuccess = $true
    } else {
        Write-Warning "XGBoost config partially updated ($updateCount/$($updates.Count) parameters)"
        $xgbSuccess = $false
    }
}


# RESUMO

Write-Header "`n================================================================="

if ($DryRun) {
    Write-Host "  [DRY-RUN] No files were modified" -ForegroundColor Yellow
    Write-Info "Run without -DryRun flag to apply changes"
} else {
    if ($mlpSuccess -and $xgbSuccess) {
        Write-Host "  [SUCCESS] All hyperparameters synchronized!" -ForegroundColor Green
        Write-Info "Federated configs are now aligned with centralized training"
    } elseif ($mlpSuccess -or $xgbSuccess) {
        Write-Host "  [PARTIAL] Some hyperparameters synchronized" -ForegroundColor Yellow
        if ($mlpSuccess) { Write-Success "MLP synchronized" }
        if ($xgbSuccess) { Write-Success "XGBoost synchronized" }
    } else {
        Write-Host "  [FAILED] No hyperparameters synchronized" -ForegroundColor Red
        Write-Info "Train centralized models first to generate hyperparameters"
    }
}

Write-Header "=================================================================`n"

# Exit with appropriate code
if ($DryRun -or ($mlpSuccess -and $xgbSuccess)) {
    exit 0
} else {
    exit 1
}
