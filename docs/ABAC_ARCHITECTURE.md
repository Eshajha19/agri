# ABAC Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           firestore-abac.js Helper Module                 │  │
│  │  • getCurrentUserOrgId() - Get user's organization        │  │
│  │  • createDocWithOrg() - Auto-inject organizationId        │  │
│  │  • queryOrgDocs() - Query org-scoped data                 │  │
│  │  • canAccessDoc() - Check access permissions              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Firebase SDK
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Firebase Auth                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ID Token with Custom Claims:                                    │
│  {                                                                │
│    "uid": "user123",                                              │
│    "role": "farmer",              ← Set by backend               │
│    "organizationId": "farm_abc"   ← Set by backend               │
│  }                                                                │
│                                                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Token validation
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Firestore Security Rules                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  function canAccessOrg(docOrgId) {                               │
│    let userOrgId = request.auth.token.organizationId;            │
│    return userOrgId == docOrgId || hasRole('admin');             │
│  }                                                                │
│                                                                   │
│  match /posts/{postId} {                                         │
│    allow read: if canAccessOrg(resource.data.organizationId);    │
│    allow create: if hasValidOrgId();                             │
│  }                                                                │
│                                                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Access granted/denied
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Firestore Database                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Collection: posts                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Document: post_001                                          │ │
│  │ {                                                            │ │
│  │   "userId": "user123",                                       │ │
│  │   "organizationId": "farm_abc",  ← Required for isolation   │ │
│  │   "content": "Great harvest!",                               │ │
│  │   "createdAt": "2026-05-13T10:00:00Z"                        │ │
│  │ }                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Collection: users                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Document: user123                                            │ │
│  │ {                                                            │ │
│  │   "role": "farmer",                                          │ │
│  │   "organizationId": "farm_abc",  ← Fallback for rules       │ │
│  │   "email": "farmer@example.com"                              │ │
│  │ }                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Admin SDK
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           auth_claims.py Module                           │  │
│  │  • set_user_claims() - Set role & organizationId          │  │
│  │  • update_user_role() - Change user role                  │  │
│  │  • update_user_organization() - Move user to new org      │  │
│  │  • sync_claims_from_firestore() - Sync claims             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  API Endpoints:                                                   │
│  • POST /api/auth/register - Create user with claims             │
│  • POST /api/admin/update-role - Update user role                │
│  • POST /api/admin/move-user - Move user to different org        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow: Creating a Post

```
┌──────────┐
│  User    │
│ (farmer) │
└────┬─────┘
     │
     │ 1. Click "Create Post"
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Frontend: createDocWithOrg('posts', { content })      │
└────┬───────────────────────────────────────────────────┘
     │
     │ 2. Get user's organizationId
     │    - Check cache (5 min TTL)
     │    - Check token claims
     │    - Fallback to Firestore
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Prepare document:                                      │
│  {                                                       │
│    userId: "user123",                                    │
│    organizationId: "farm_abc",  ← Auto-injected         │
│    content: "Great harvest!",                            │
│    createdAt: serverTimestamp()                          │
│  }                                                       │
└────┬───────────────────────────────────────────────────┘
     │
     │ 3. Send to Firestore
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Firestore Rules: Validate                              │
│  ✓ User is authenticated                                │
│  ✓ userId matches auth.uid                              │
│  ✓ organizationId matches user's org                    │
│  ✓ Content is valid                                     │
└────┬───────────────────────────────────────────────────┘
     │
     │ 4. Access granted
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Firestore: Document created                            │
│  posts/post_001                                          │
└────┬───────────────────────────────────────────────────┘
     │
     │ 5. Success response
     │
     ↓
┌──────────┐
│  User    │
│ sees post│
└──────────┘
```

## Data Flow: Reading Posts

```
┌──────────┐
│  User A  │
│ (org_a)  │
└────┬─────┘
     │
     │ 1. Load posts
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Frontend: queryOrgDocs('posts')                        │
└────┬───────────────────────────────────────────────────┘
     │
     │ 2. Get user's organizationId
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Query Firestore:                                       │
│  collection('posts')                                     │
│    .where('organizationId', '==', 'org_a')              │
│    .orderBy('createdAt', 'desc')                         │
└────┬───────────────────────────────────────────────────┘
     │
     │ 3. Send query
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  Firestore Rules: Validate each document                │
│                                                          │
│  Post 1: organizationId = "org_a"                       │
│  ✓ User's org matches → ALLOW                           │
│                                                          │
│  Post 2: organizationId = "org_b"                       │
│  ✗ User's org doesn't match → DENY                      │
│                                                          │
│  Post 3: organizationId = "org_a"                       │
│  ✓ User's org matches → ALLOW                           │
└────┬───────────────────────────────────────────────────┘
     │
     │ 4. Return filtered results
     │
     ↓
┌────────────────────────────────────────────────────────┐
│  User A sees only posts from org_a                      │
│  • Post 1 ✓                                             │
│  • Post 3 ✓                                             │
│  (Post 2 from org_b is hidden)                          │
└─────────────────────────────────────────────────────────┘
```

## Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: Frontend                         │
│  • Input validation                                          │
│  • UI-level access control                                   │
│  • Auto-inject organizationId                                │
│  ⚠️  NOT TRUSTED - Can be bypassed                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                Layer 2: Firebase Auth                        │
│  • Token verification                                        │
│  • Custom claims (cryptographically signed)                  │
│  • Cannot be tampered by client                              │
│  ✓ TRUSTED                                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            Layer 3: Firestore Security Rules                 │
│  • Validate organizationId match                             │
│  • Enforce immutability                                      │
│  • Check role permissions                                    │
│  • Fail-closed design                                        │
│  ✓ TRUSTED - Server-side enforcement                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                Layer 4: Firestore Database                   │
│  • Data at rest encryption                                   │
│  • Backup and recovery                                       │
│  • Audit logging                                             │
│  ✓ TRUSTED                                                   │
└─────────────────────────────────────────────────────────────┘
```

## Custom Claims Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    User Registration                          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Backend: POST /api/auth/register                            │
│  {                                                            │
│    email: "farmer@example.com",                               │
│    password: "***",                                           │
│    invite_code: "FARM_ABC_INVITE"                             │
│  }                                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  1. Create Firebase Auth user                                │
│     firebase_auth.create_user(email, password)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Determine organization from invite code                  │
│     org_id = lookup_invite_code("FARM_ABC_INVITE")           │
│     → "farm_abc"                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Set custom claims on auth token                          │
│     set_user_claims(uid, role='farmer', org_id='farm_abc')   │
│                                                               │
│     Firebase Auth Token now contains:                         │
│     {                                                         │
│       "uid": "user123",                                       │
│       "role": "farmer",        ← Custom claim                 │
│       "organizationId": "farm_abc"  ← Custom claim            │
│     }                                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Create Firestore user profile (fallback)                 │
│     db.collection('users').doc(uid).set({                     │
│       role: 'farmer',                                         │
│       organizationId: 'farm_abc',                             │
│       email: 'farmer@example.com'                             │
│     })                                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  5. User can now access only farm_abc data                   │
└─────────────────────────────────────────────────────────────┘
```

## Organization Isolation Example

```
Database State:
┌─────────────────────────────────────────────────────────────┐
│  posts collection                                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ post_001                                             │    │
│  │ organizationId: "farm_abc"                           │    │
│  │ content: "Great harvest!"                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ post_002                                             │    │
│  │ organizationId: "farm_xyz"                           │    │
│  │ content: "Need pest advice"                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ post_003                                             │    │
│  │ organizationId: "farm_abc"                           │    │
│  │ content: "Selling equipment"                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

User Access:
┌─────────────────────────────────────────────────────────────┐
│  User A (organizationId: "farm_abc", role: "farmer")        │
│  Can see:                                                    │
│  ✓ post_001 (same org)                                      │
│  ✗ post_002 (different org)                                 │
│  ✓ post_003 (same org)                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  User B (organizationId: "farm_xyz", role: "farmer")        │
│  Can see:                                                    │
│  ✗ post_001 (different org)                                 │
│  ✓ post_002 (same org)                                      │
│  ✗ post_003 (different org)                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Admin (organizationId: "farm_abc", role: "admin")          │
│  Can see:                                                    │
│  ✓ post_001 (admin override)                                │
│  ✓ post_002 (admin override)                                │
│  ✓ post_003 (admin override)                                │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimization

```
Request Flow with Caching:

First Request:
┌──────────┐
│  User    │
└────┬─────┘
     │ 1. Request posts
     ↓
┌─────────────────────────┐
│  Check cache             │
│  ✗ Cache miss            │
└────┬────────────────────┘
     │ 2. Get token
     ↓
┌─────────────────────────┐
│  Get ID token            │
│  (includes claims)       │
└────┬────────────────────┘
     │ 3. Extract orgId
     ↓
┌─────────────────────────┐
│  Cache organizationId    │
│  TTL: 5 minutes          │
└────┬────────────────────┘
     │ 4. Query Firestore
     ↓
┌─────────────────────────┐
│  Return results          │
└─────────────────────────┘

Subsequent Requests (within 5 min):
┌──────────┐
│  User    │
└────┬─────┘
     │ 1. Request posts
     ↓
┌─────────────────────────┐
│  Check cache             │
│  ✓ Cache hit             │
│  (no token fetch needed) │
└────┬────────────────────┘
     │ 2. Query Firestore
     ↓
┌─────────────────────────┐
│  Return results          │
└─────────────────────────┘

Performance Gain:
• Eliminates 1 token fetch per request
• Reduces latency by ~50-100ms
• Reduces Firebase Auth API calls
```

## Migration Process

```
Before Migration:
┌─────────────────────────────────────────────────────────────┐
│  posts collection                                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │ post_001                                             │    │
│  │ userId: "user123"                                    │    │
│  │ content: "Hello"                                     │    │
│  │ ⚠️  Missing organizationId                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

Migration Script:
┌─────────────────────────────────────────────────────────────┐
│  python migration_add_org_id.py                              │
│                                                               │
│  1. Read post_001                                             │
│  2. Get userId: "user123"                                     │
│  3. Lookup user's organizationId: "farm_abc"                  │
│  4. Update post_001 with organizationId                       │
└─────────────────────────────────────────────────────────────┘

After Migration:
┌─────────────────────────────────────────────────────────────┐
│  posts collection                                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │ post_001                                             │    │
│  │ userId: "user123"                                    │    │
│  │ organizationId: "farm_abc"  ← Added                  │    │
│  │ content: "Hello"                                     │    │
│  │ migratedAt: "2026-05-13T10:00:00Z"                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Defense in Depth**: Multiple security layers (frontend, auth, rules, database)
2. **Fail-Closed**: Access denied by default unless explicitly granted
3. **Immutable Tenancy**: Users cannot switch organizations
4. **Performance**: Custom claims reduce Firestore reads
5. **Auditability**: All access is logged and traceable
6. **Scalability**: Indexed queries perform well at scale
