@echo off
REM Firestore Emulator Setup Script (Windows)
REM Usage: setup_firestore_emulator.bat

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Firestore Emulator Setup Script (Windows)
echo ==========================================
echo.

REM Check if Firebase CLI is installed
firebase --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Firebase CLI is not installed
    echo Please install it with: npm install -g firebase-tools
    exit /b 1
)

echo [OK] Firebase CLI found

REM Check if we're in the project root
if not exist "firestore.rules" (
    echo X firestore.rules not found in current directory
    echo Please run this script from the project root
    exit /b 1
)

echo [OK] firestore.rules found

REM Create firebase.json if it doesn't exist
if not exist "firebase.json" (
    echo Creating firebase.json...
    (
        echo {
        echo   "firestore": {
        echo     "rules": "firestore.rules",
        echo     "indexes": "firestore.indexes.json"
        echo   },
        echo   "emulators": {
        echo     "firestore": {
        echo       "host": "localhost",
        echo       "port": 8080
        echo     },
        echo     "ui": {
        echo       "enabled": true,
        echo       "host": "localhost",
        echo       "port": 4000
        echo     }
        echo   }
        echo }
    ) > firebase.json
    echo [OK] firebase.json created
) else (
    echo [OK] firebase.json already exists
)

REM Download emulators
echo.
echo Downloading Firestore emulator...
call firebase setup:emulators:firestore

echo.
echo ==========================================
echo [OK] Setup Complete!
echo ==========================================
echo.
echo To start the emulator, run:
echo   firebase emulators:start
echo.
echo To run tests with the emulator, run:
echo   set FIRESTORE_EMULATOR_HOST=localhost:8080
echo   pytest tests/test_firestore_rules.py -v
echo.
echo The Firestore UI will be available at:
echo   http://localhost:4000
echo.
