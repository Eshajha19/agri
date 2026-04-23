import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', 'dev-dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
      'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
    },
  },
])

/**
 * CSS Best Practices Guide (Manual Enforcement):
 * 
 * 1. Avoid !important - Use proper CSS specificity instead
 *    - Bad:  .button { color: blue !important; }
 *    - Good: .navbar .button { color: blue; }
 * 
 * 2. Use CSS Custom Properties for theming
 *    - Bad:  .theme { color: #000 !important; }
 *    - Good: :root.theme { --text-color: #000; } and use var(--text-color)
 * 
 * 3. Organize selectors by specificity (low to high)
 *    - Start with base styles
 *    - Add component-level overrides
 *    - Use theme-specific classes last (with proper specificity, not !important)
 * 
 * 4. Reference: themes/sunlight.css demonstrates proper theme implementation
 *    without relying on !important
 * 
 * 5. For CSS linting, use stylelint (https://stylelint.io/)
 *    Configuration: .stylelintrc.json in project root (optional, for future enhancement)
 */
