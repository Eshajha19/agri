#!/bin/bash
# Firestore Emulator Setup Script
# Usage: bash scripts/setup_firestore_emulator.sh

set -e

echo "=========================================="
echo "Firestore Emulator Setup Script"
echo "=========================================="

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI is not installed"
    echo "Please install it with: npm install -g firebase-tools"
    exit 1
fi

echo "✅ Firebase CLI found"

# Check if we're in the project root
if [ ! -f "firestore.rules" ]; then
    echo "❌ firestore.rules not found in current directory"
    echo "Please run this script from the project root"
    exit 1
fi

echo "✅ firestore.rules found"

# Create firebase.json if it doesn't exist
if [ ! -f "firebase.json" ]; then
    echo "📝 Creating firebase.json..."
    cat > firebase.json << 'EOF'
{
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "emulators": {
    "firestore": {
      "host": "localhost",
      "port": 8080
    },
    "ui": {
      "enabled": true,
      "host": "localhost",
      "port": 4000
    }
  }
}
EOF
    echo "✅ firebase.json created"
else
    echo "✅ firebase.json already exists"
fi

# Download emulators
echo "📥 Downloading Firestore emulator..."
firebase setup:emulators:firestore

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the emulator, run:"
echo "  firebase emulators:start"
echo ""
echo "To run tests with the emulator, run:"
echo "  FIRESTORE_EMULATOR_HOST=localhost:8080 pytest tests/test_firestore_rules.py -v"
echo ""
echo "The Firestore UI will be available at:"
echo "  http://localhost:4000"
echo ""
