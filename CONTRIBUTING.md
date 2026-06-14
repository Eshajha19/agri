# Contributing to Fasal Saathi

Thank you for your interest in contributing to Fasal Saathi — an open-source platform built to help farmers with smart agricultural solutions. We welcome all contributions: features, bug fixes, UI improvements, and documentation.

---

## How to Contribute

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

Use descriptive branch names:
- `fix/navbar-mobile-overflow`
- `feat/add-loading-skeleton`
- `docs/improve-installation-guide`
- `refactor/extract-modal-component`

### 4. Install Dependencies

**Frontend:**
```bash
cd frontend
npm install
```

**Backend (Python):**
```bash
cd ..
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Environment Setup

1. Copy `.env.example` to `.env` in the frontend directory
2. Copy the root `.env.example` to `.env` for backend configuration
3. Configure your Firebase project credentials
4. Set up a Weather API key (OpenWeatherMap)

### 6. Run the Project

**Frontend:**
```bash
cd frontend
npm run dev
```
The frontend starts at `http://localhost:5173`.

**Backend (FastAPI):**
```bash
python -m uvicorn main:app --reload --port 8000
```
The backend starts at `http://localhost:8000`.

### 7. Make Your Changes

- Keep code clean and readable
- Follow existing project structure
- Ensure responsiveness for UI changes
- Test your changes before submitting
- Follow the coding standards below

### 8. Commit Your Changes

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
- `perf:` for performance improvements
- `test:` for adding or updating tests

### 9. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with a clear description of your changes.

---

## Code Guidelines

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
- Test in Light, Dark, and Night Light themes

### Accessibility

- Use semantic HTML elements
- Add ARIA labels where appropriate
- Ensure keyboard navigation works
- Test with screen reader announcements
- Use `aria-hidden` on decorative icons

---

## Testing

### Frontend
```bash
cd frontend
npm run build    # Verify build succeeds
npm run lint     # Check for lint errors
```

### Backend
```bash
python -m pytest tests/
```

---

## Reporting Issues

If you find a bug:

1. Open an issue
2. Clearly describe the problem
3. Add steps to reproduce
4. Include screenshots (if possible)
5. Mention your environment (OS, browser, Node version)

---

## Issue Rules

**Before starting work:**

1. Comment "assign me" on the issue
2. Wait for assignment before starting work
3. Include `NSoC'26` tag in your PR title if participating in NSoC

---

## Development Setup Summary

```bash
# Clone and setup
git clone https://github.com/<your-username>/agri.git
cd agri
cd frontend && npm install && cd ..
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Start frontend (terminal 1)
cd frontend && npm run dev

# Start backend (terminal 2)
python -m uvicorn main:app --reload --port 8000
```

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
