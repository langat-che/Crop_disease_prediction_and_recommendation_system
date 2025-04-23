@echo off
REM Full automation script for crop classification workflow
echo --- Starting Full Automated Workflow (Initial Train + Fine-Tune) ---
setlocal enabledelayedexpansion

REM --- Configuration ---
set "DATA_BASE_DIR=Data"
set "MODEL_SAVE_DIR=models"
set "REPORTS_DIR=reports"
set "HISTORY_SAVE_DIR=training_history"
set "SCRIPTS_DIR=Scripts"

REM --- Initial Training Parameters ---
set "IMG_HEIGHT=160"
set "IMG_WIDTH=160"
set "BATCH_SIZE=8"
set "INITIAL_EPOCHS=15"
set "INITIAL_LR=0.001"
set "DROPOUT_RATE=0.4"
set "AUGMENTATION=mild"
set "INITIAL_EARLY_STOPPING=5"

REM --- Fine-Tuning Parameters ---
set "FINE_TUNE_EPOCHS=20"
set "FINE_TUNE_LR=5e-06"
set "FINE_TUNE_AT=100"
set "FT_EARLY_STOPPING=7"

REM --- Ensure output directories exist ---
if not exist "%MODEL_SAVE_DIR%" mkdir "%MODEL_SAVE_DIR%"
if not exist "%REPORTS_DIR%" mkdir "%REPORTS_DIR%"
if not exist "%HISTORY_SAVE_DIR%" mkdir "%HISTORY_SAVE_DIR%"

REM --- Main Loop ---
for %%C in (maize onion tomato) do (
    echo.
    echo ^>^>^> ================================================== ^<^<^<
    echo ^>^>^> Processing Crop: %%C ^<^<^<
    echo ^>^>^> ================================================== ^<^<^<
    set "CROP_TYPE=%%C"
    set "SKIP_CROP=0"

    REM --- Construct expected initial model filename ---
    set "INITIAL_MODEL_FILENAME=!CROP_TYPE!_initial_best_img%IMG_HEIGHT%_dr%DROPOUT_RATE%_lr%INITIAL_LR%_aug_%AUGMENTATION%.keras"
    set "INITIAL_MODEL_PATH=%MODEL_SAVE_DIR%\!INITIAL_MODEL_FILENAME!"

    REM --- Step 2: Initial Training ---
    echo.
    echo --- Step 2: Running Initial Training for !CROP_TYPE! ---
    python "%SCRIPTS_DIR%\train.py" ^
        --crop_type "!CROP_TYPE!" ^
        --data_base_dir "%DATA_BASE_DIR%" ^
        --epochs "%INITIAL_EPOCHS%" ^
        --dropout_rate "%DROPOUT_RATE%" ^
        --model_save_dir "%MODEL_SAVE_DIR%" ^
        --img_height "%IMG_HEIGHT%" ^
        --img_width "%IMG_WIDTH%" ^
        --batch_size "%BATCH_SIZE%" ^
        --learning_rate "%INITIAL_LR%" ^
        --augmentation_strength "%AUGMENTATION%" ^
        --history_save_path "%HISTORY_SAVE_DIR%" ^
        --early_stopping_patience "%INITIAL_EARLY_STOPPING%" ^
        --verbose 1 > "%REPORTS_DIR%\!CROP_TYPE!_train_log.txt" 2>&1
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Initial training for !CROP_TYPE! failed. Check '%REPORTS_DIR%\!CROP_TYPE!_train_log.txt' for details.
        set "SKIP_CROP=1"
    )

    REM --- Step 3: Evaluate Initial Model ---
    echo.
    echo --- Step 3: Evaluating Best Initial Model for !CROP_TYPE! ---
    if exist "!INITIAL_MODEL_PATH!" (
        python "%SCRIPTS_DIR%\evaluate.py" ^
            --model_path "!INITIAL_MODEL_PATH!" ^
            --crop_type "!CROP_TYPE!" ^
            --data_base_dir "%DATA_BASE_DIR%" ^
            --img_height "%IMG_HEIGHT%" ^
            --img_width "%IMG_WIDTH%" ^
            --batch_size "%BATCH_SIZE%" ^
            --report_save_path "%REPORTS_DIR%" ^
            --cm_save_path "%REPORTS_DIR%" ^
            --verbose 0 > "%REPORTS_DIR%\!CROP_TYPE!_eval_initial_log.txt" 2>&1
        if !ERRORLEVEL! neq 0 (
            echo ERROR: Initial evaluation for !CROP_TYPE! failed. Check '%REPORTS_DIR%\!CROP_TYPE!_eval_initial_log.txt' for details.
            set "SKIP_CROP=1"
        )
    ) else (
        echo WARNING: Initial model not found at '!INITIAL_MODEL_PATH!'. Cannot evaluate or fine-tune.
        set "SKIP_CROP=1"
    )

    if "!SKIP_CROP!"=="0" (
        REM --- Step 4: Fine-Tuning ---
        echo.
        echo --- Step 4: Running Fine-Tuning for !CROP_TYPE! ---
        for %%F in ("!INITIAL_MODEL_PATH!") do set "INITIAL_MODEL_STEM=%%~nF"
        set "FT_MODEL_FILENAME=!CROP_TYPE!_finetuned_L%FINE_TUNE_AT%_ftlr%FINE_TUNE_LR%_from_!INITIAL_MODEL_STEM!.keras"
        set "FT_MODEL_PATH=%MODEL_SAVE_DIR%\!FT_MODEL_FILENAME!"

        python "%SCRIPTS_DIR%\fine_tune.py" ^
            --initial_model_path "!INITIAL_MODEL_PATH!" ^
            --crop_type "!CROP_TYPE!" ^
            --data_base_dir "%DATA_BASE_DIR%" ^
            --fine_tune_epochs "%FINE_TUNE_EPOCHS%" ^
            --fine_tune_lr "%FINE_TUNE_LR%" ^
            --fine_tune_at "%FINE_TUNE_AT%" ^
            --model_save_dir "%MODEL_SAVE_DIR%" ^
            --img_height "%IMG_HEIGHT%" ^
            --img_width "%IMG_WIDTH%" ^
            --batch_size "%BATCH_SIZE%" ^
            --augmentation_strength "%AUGMENTATION%" ^
            --history_save_path "%HISTORY_SAVE_DIR%" ^
            --early_stopping_patience "%FT_EARLY_STOPPING%" ^
            --verbose 1 > "%REPORTS_DIR%\!CROP_TYPE!_fine_tune_log.txt" 2>&1
        if !ERRORLEVEL! neq 0 (
            echo ERROR: Fine-tuning for !CROP_TYPE! failed. Check '%REPORTS_DIR%\!CROP_TYPE!_fine_tune_log.txt' for details.
        )

        REM --- Step 5: Evaluate Fine-Tuned Model ---
        echo.
        echo --- Step 5: Evaluating Best Fine-Tuned Model for !CROP_TYPE! ---
        if exist "!FT_MODEL_PATH!" (
            python "%SCRIPTS_DIR%\evaluate.py" ^
                --model_path "!FT_MODEL_PATH!" ^
                --crop_type "!CROP_TYPE!" ^
                --data_base_dir "%DATA_BASE_DIR%" ^
                --img_height "%IMG_HEIGHT%" ^
                --img_width "%IMG_WIDTH%" ^
                --batch_size "%BATCH_SIZE%" ^
                --report_save_path "%REPORTS_DIR%" ^
                --cm_save_path "%REPORTS_DIR%" ^
                --verbose 0 > "%REPORTS_DIR%\!CROP_TYPE!_eval_finetuned_log.txt" 2>&1
            if !ERRORLEVEL! neq 0 (
                echo ERROR: Fine-tuned evaluation for !CROP_TYPE! failed. Check '%REPORTS_DIR%\!CROP_TYPE!_eval_finetuned_log.txt' for details.
            )
        ) else (
            echo WARNING: Fine-tuned model not found at '!FT_MODEL_PATH!'. Skipping evaluation.
        )
    )
)

echo.
echo --- Full Automated Workflow Finished ---
endlocal