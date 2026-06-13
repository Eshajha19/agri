# Firebase Authentication Security Audit

## 1. Authorized Domains

| Domain | Used For | Status |
|--------|----------|--------|
| `fasal-saathi.firebaseapp.com` | Firebase Auth hosted UI, OAuth redirect | ✅ Expected |
| `*.web.app` | Firebase Hosting preview channels | ✅ In CSP frame-src |
| Custom domain (Vercel) | Production frontend | ⚠️ Must be added in Firebase Console |

**Risk**: If the production domain is not listed in Firebase Console
( Authentication → Settings → Authorized domains), sign-in redirects
will fail with `auth/unauthorized-domain`.

**Action**: Verify `fasalsaathi.agri` (or the actual domain) is listed.

---

## 2. OAuth Redirect URIs

| Provider | Redirect URI | Status |
|----------|-------------|--------|
| Google | `https://fasal-saathi.firebaseapp.com/__/auth/handler` | ✅ Firebase default |
| Google | `https://{custom-domain}/__/auth/handler` | ⚠️ Must be added |

Firebase Auth uses a single redirect URI pattern for OAuth callbacks:
`https://<project>.firebaseapp.com/__/auth/handler`.  Custom domains
must have this URI registered in the Google Cloud Console.

**Action**: Add the custom domain redirect URI to
Google Cloud Console → APIs & Services → Credentials.

---

## 3. CSP Allowlist

| Directive | Current Entry | Required For |
|-----------|--------------|--------------|
| `frame-src` | `*.firebaseapp.com` | Auth hidden iframe (cross-tab sync) |
| `frame-src` | `*.web.app` | Preview channel Auth |
| `frame-src` | `accounts.google.com` | Google Sign-In popup frame |
| `frame-src` | `apis.google.com` | Google Sign-In SDK frame |
| `connect-src` | `identitytoolkit.googleapis.com` | Firebase Auth REST API |
| `connect-src` | `securetoken.googleapis.com` | Token refresh endpoint |
| `connect-src` | `www.googleapis.com` | Google API calls |
| `connect-src` | `*.googleapis.com` | Additional Google APIs |
| `script-src` | `apis.google.com` | Google Sign-In JS SDK |

**Risk**: Missing entries in `connect-src` can cause silent token refresh
failures.  Verify `securetoken.googleapis.com` is included.

---

## 4. Popup Handling Flow

```
User clicks "Sign in with Google"
  → signInWithPopup(auth, provider)
     ├── Success → syncUserToFirestore → navigate
     └── Failure → fallback to signInWithRedirect(provider)
                     ├── User leaves to Google
                     └── Returns → getRedirectResult on mount
                                    └── syncUserToFirestore → navigate
```

**COOP requirement**: `Cross-Origin-Opener-Policy: same-origin-allow-popups`
is required for `signInWithPopup` to detect `window.closed` and receive
`postMessage` from the Google OAuth popup.

**Risk**: Without the COOP header, `signInWithPopup` hangs indefinitely
after the user authenticates, forcing a fallback to redirect which
adds a full page reload.

---

## 5. Token Refresh Flow

Firebase Auth automatically refreshes the ID token every hour via
`POST https://securetoken.googleapis.com/v1/token`.

```
Client SDK
  → POST securetoken.googleapis.com (refresh_token)
  ← { id_token, access_token, expires_in }
  → SDK attaches new id_token to subsequent requests
```

**CSP requirement**: `connect-src` must include `securetoken.googleapis.com`.
If missing, the browser blocks the fetch request and the SDK throws an
uncaught error, silently breaking authentication after the first token
expiry.

**Backend verification**: The server verifies `id_token` via Firebase
Admin SDK's `verify_id_token()`.  No additional configuration needed
as long as the token reaches the backend via the `Authorization` header.

---

## Summary of Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Custom domain not in Firebase Console authorized domains | 🔴 High | Add to Firebase Console → Authentication → Settings |
| 2 | OAuth redirect URI not registered for custom domain | 🟡 Medium | Add to Google Cloud Console credentials |
| 3 | `securetoken.googleapis.com` missing from CSP connect-src | 🔴 High (silent) | Include in connect-src directive |
| 4 | COOP header missing or set to same-origin | 🟡 Medium | Set to same-origin-allow-popups |
| 5 | Popup failure fallback to redirect increases latency | 🟢 Low | Acceptable — already implemented |
