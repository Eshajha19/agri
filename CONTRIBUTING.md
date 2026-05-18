# 🤝 Contributing to Fasal Saathi

Thank you for your interest in contributing to Fasal Saathi 🌾 — an open-source platform built to help farmers with smart agricultural solutions. We welcome all contributions: features, bug fixes, UI improvements, and documentation.

---

## 🚀 How to Contribute

### 1. Fork the Repository

Click the **Fork** button at the top right of the repository.

### 2. Clone Your Fork

```bash
git clone https://github.com/<your-username>/agri.git
cd agri
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Install Dependencies

```bash
npm install
```

### 5. Run the Project

**Frontend:**
```bash
npm run dev
```

**Backend (FastAPI):**
```bash
python -m uvicorn main:app --reload --port 8000
```

### 6. Make Your Changes

- Keep code clean and readable
- Follow existing project structure
- Ensure responsiveness for UI changes
- Test your changes before submitting

### 7. Commit Your Changes

```bash
git add .
git commit -m "type: short description of changes"
```

Use conventional commit messages:
- `fix:` for bug fixes
- `feat:` for new features
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code restructuring

### 8. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with a clear description of your changes.

---

## 🧠 Code Guidelines

### Clean Code Practices

- Use meaningful variable and function names
- Keep functions small and focused
- Avoid duplicate code
- Follow existing naming conventions

### Comments

- Add comments for complex logic only
- Keep code self-explanatory
- Use JSDoc for function documentation when applicable

### UI Guidelines

- Maintain design consistency
- Ensure mobile responsiveness
- Avoid breaking existing layout
- Test across different screen sizes

---

## 🐛 Reporting Issues

If you find a bug:

1. Open an issue
2. Clearly describe the problem
3. Add steps to reproduce
4. Include screenshots (if possible)
5. Mention your environment (OS, browser, Node version)

---

## 🏷️ Issue Rules

**Before starting work:**

1. Comment "assign me" on the issue
2. Wait for assignment before starting work
3. Include `NSoC'26` tag in your PR title if participating in NSoC

---

## 📦 Development Setup Summary

```bash
# Install dependencies
npm install

# Start frontend
npm run dev

# Start backend (in separate terminal)
python -m uvicorn main:app --reload --port 8000
```

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
