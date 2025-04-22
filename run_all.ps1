# PowerShell Automation Script for Crop Classification Workflow
Write-Host "--- Starting Full Automated Workflow (Initial Train + Fine-Tune) ---" -ForegroundColor Yellow

# --- Configuration ---
$DataBaseDir = "Data"
$ModelSaveDir = "models"
$ReportsDir = "reports"
$HistorySaveDir = "training_history"
$ScriptsDir = "Scripts"

# --- Initial Training Parameters ---
$ImgHeight = 160
$ImgWidth = 160
$BatchSize = 16 # Adjusted batch size
$InitialEpochs = 15
$InitialLR = 0.001
$DropoutRate = 0.4
$Augmentation = 'mild'
$InitialEarlyStopping = 5

# --- Fine-Tuning Parameters ---
$FineTuneEpochs = 20
$FineTuneLR = 5e-6 # PowerShell understands scientific notation
$FineTuneAt = 100
$FTEarlyStopping = 7

# --- Ensure output directories exist ---
# Use Join-Path for cross-platform compatibility (though running on Windows here)
$ModelSavePath = Join-Path $PSScriptRoot $ModelSaveDir
$ReportsPath = Join-Path $PSScriptRoot $ReportsDir
$HistorySavePath = Join-Path $PSScriptRoot $HistorySaveDir
If (!(Test-Path $ModelSavePath)) { New-Item -ItemType Directory -Force -Path $ModelSavePath | Out-Null }
If (!(Test-Path $ReportsPath)) { New-Item -ItemType Directory -Force -Path $ReportsPath | Out-Null }
If (!(Test-Path $HistorySavePath)) { New-Item -ItemType Directory -Force -Path $HistorySavePath | Out-Null }

# --- Main Loop ---
$Crops = @("maize", "onion", "tomato")

foreach ($CropType in $Crops) {
    Write-Host ""
    Write-Host "##################################################" -ForegroundColor Cyan
    Write-Host "### Processing Crop: $($CropType.ToUpper())" -ForegroundColor Cyan
    Write-Host "##################################################" -ForegroundColor Cyan

    # --- Construct expected initial model filename ---
    # PowerShell formatting is more flexible
    $LRString = "{0:G}" -f $InitialLR # General format for float
    $InitialModelFilename = "${CropType}_initial_best_img${ImgHeight}_dr${DropoutRate}_lr${LRString}_aug_${Augmentation}.keras"
    $InitialModelPath = Join-Path $ModelSavePath $InitialModelFilename

    # --- Step 1: Data Cleaning (Optional - Uncomment to run) ---
    # Write-Host ""
    # Write-Host "--- Step 1: Cleaning Data for $CropType (Check disabled) ---"
    # $CleanArgs = @(
    #     "--data_base_dir", $DataBaseDir,
    #     "--crop_type", $CropType,
    #     "--check_corruption" # Add other flags like --fix_filenames if needed
    # )
    # python (Join-Path $ScriptsDir "clean_data.py") @CleanArgs

    # --- Step 2: Initial Training ---
    Write-Host ""
    Write-Host "--- Step 2: Running Initial Training for $CropType ---"
    $TrainArgs = @(
        "--crop_type", $CropType,
        "--data_base_dir", $DataBaseDir,
        "--epochs", $InitialEpochs,
        "--dropout_rate", $DropoutRate,
        "--model_save_dir", $ModelSaveDir,
        "--img_height", $ImgHeight,
        "--img_width", $ImgWidth,
        "--batch_size", $BatchSize,
        "--learning_rate", $InitialLR,
        "--augmentation_strength", $Augmentation,
        "--history_save_path", $HistorySaveDir,
        "--early_stopping_patience", $InitialEarlyStopping,
        "--verbose", 1
    )
    # Use '& python' or 'python.exe' if python is not directly in PATH in PowerShell session
    python (Join-Path $ScriptsDir "train.py") @TrainArgs
    $TrainExitCode = $LASTEXITCODE

    # --- Step 3: Evaluate Initial Model ---
    Write-Host ""
    Write-Host "--- Step 3: Evaluating Best Initial Model for $CropType ---"
    if (Test-Path $InitialModelPath -PathType Leaf) {
        Write-Host "  Found initial model: $InitialModelPath"
        $EvalArgs1 = @(
            "--model_path", $InitialModelPath,
            "--crop_type", $CropType,
            "--data_base_dir", $DataBaseDir,
            "--img_height", $ImgHeight,
            "--img_width", $ImgWidth,
            "--batch_size", $BatchSize,
            "--report_save_path", $ReportsDir,
            "--cm_save_path", $ReportsDir,
            "--verbose", 0
        )
        python (Join-Path $ScriptsDir "evaluate.py") @EvalArgs1
    } else {
        Write-Host "WARNING: Initial model not found at '$InitialModelPath'. Cannot evaluate or fine-tune." -ForegroundColor Yellow
        continue # Skip to the next crop
    }

    # --- Step 4: Fine-Tuning ---
    Write-Host ""
    Write-Host "--- Step 4: Running Fine-Tuning for $CropType ---"
    # Construct fine-tuned model filename
    $InitialModelStem = (Get-Item $InitialModelPath).BaseName
    $FTLRString = "{0:G}" -f $FineTuneLR # General format, might output 5e-06
    $FTModelFilename = "${CropType}_finetuned_L${FineTuneAt}_ftlr${FTLRString}_from_${InitialModelStem}.keras"
    $FTModelPath = Join-Path $ModelSavePath $FTModelFilename

    $FTArgs = @(
        "--initial_model_path", $InitialModelPath,
        "--crop_type", $CropType,
        "--data_base_dir", $DataBaseDir,
        "--fine_tune_epochs", $FineTuneEpochs,
        "--fine_tune_lr", $FineTuneLR,
        "--fine_tune_at", $FineTuneAt,
        "--model_save_dir", $ModelSaveDir,
        "--img_height", $ImgHeight,
        "--img_width", $ImgWidth,
        "--batch_size", $BatchSize,
        "--augmentation_strength", $Augmentation,
        "--history_save_path", $HistorySaveDir,
        "--early_stopping_patience", $FTEarlyStopping,
        "--verbose", 1
    )
    python (Join-Path $ScriptsDir "fine_tune.py") @FTArgs
    $FTExitCode = $LASTEXITCODE

    # --- Step 5: Evaluate Fine-Tuned Model ---
    Write-Host ""
    Write-Host "--- Step 5: Evaluating Best Fine-Tuned Model for $CropType ---"
    if (Test-Path $FTModelPath -PathType Leaf) {
         Write-Host "  Found fine-tuned model: $FTModelPath"
        $EvalArgs2 = @(
            "--model_path", $FTModelPath,
            "--crop_type", $CropType,
            "--data_base_dir", $DataBaseDir,
            "--img_height", $ImgHeight,
            "--img_width", $ImgWidth,
            "--batch_size", $BatchSize,
            "--report_save_path", $ReportsDir,
            "--cm_save_path", $ReportsDir,
            "--verbose", 0
        )
        python (Join-Path $ScriptsDir "evaluate.py") @EvalArgs2
    } else {
        Write-Host "WARNING: Fine-tuned model not found at '$FTModelPath'. Skipping evaluation." -ForegroundColor Yellow
        Write-Host "         (This might happen if fine-tuning errored or didn't save a better model)."
    }
} # End of FOR loop

Write-Host ""
Write-Host "--- Full Automated Workflow Finished ---" -ForegroundColor Green