# Firestore Security Rules — Comprehensive Documentation

This document explains the intent, authorization model, and validation guarantees implemented in `firestore.rules`.

> Goal: make it easy to audit and safely evolve Firestore security without accidentally widening access or bypassing business logic.

---

## 1) High-level design goals

The rules in this repository are structured around the following security goals:

1. **Least privilege**
   - Every collection is restricted to a small set of roles and/or ownership rules.

2. **No implicit “default role” escalation**
   - Role checks rely on Firebase custom claims.
   - Missing/empty roles must not be treated as elevated permissions.

3. **Anti-impersonation**
   - Writes that represent user actions must bind to `request.auth.uid` (e.g., `userId`, `reporterId`).

4. **Prevent privilege escalation through self-service updates**
   - A user updating their own `/users/{uid}` profile cannot change their role to anything except `farmer`.

5. **Server-enforced rate limiting / cooldowns**
   - Cooldowns for posts, comments, and reputation changes are enforced in Firestore rules.
   - This prevents direct SDK writes from bypassing the backend/front-end rate limiter.

6. **Schema and invariants enforcement**
   - Rules validate critical fields: types, presence, lengths, and timestamp correctness.
   - Some invariants (e.g., consultation `userId` immutability) are enforced after create.

---

## 2) Authentication & authorization model

### 2.1 Role source: Firebase custom claims

Roles are evaluated using a custom claim stored in the authentication token:

- `roleClaim()` returns: `request.auth.token.get("role", "")`

Helper predicates:

- `isAuthed()`
  - `request.auth != null`

- `hasRole(role)`
  - `request.auth != null && roleClaim() == role`

- `hasAnyRole(roles)`
  - `request.auth != null && roleClaim() in roles`

- `isOwner(uid)`
  - caller must be authenticated and `request.auth.uid == uid`

### 2.2 Backend expectations

The rules assume the backend (or a role-sync job) mirrors the authoritative role into the JWT custom claims.

**Important:** The rules do not implement any “fallback role” logic.
- This avoids granting unintended access during backfill or stale JWT windows.

---

## 3) Validation guarantees implemented in rules

### 3.1 Timestamp correctness (`createdAt`)

Some write operations require:

- `request.resource.data[field] == request.time`

Currently enforced via helper:

- `hasServerTimestamp(field)`

### 3.2 Content validation

Helper validators:

- `postContentValid()`
  - `content` must be a string
  - `content.size() >= 20`

- `commentContentValid()`
  - `text` must be a string
  - `text.size() >= 5`

### 3.3 Geo/Alert field validation

For `/disaster_alerts/{alertId}`, creation requires:

- `reporterId == request.auth.uid`
- `type in ["Pest Outbreak", "Wildfire", "Extreme Weather"]`
- `severity in ["High", "Medium"]`
- `notes` is a string with `size() <= 100`
- `geohash` is a string
- `lat` and `lng` are numbers
- `createdAt` matches `request.time`

---

## 4) Cooldowns / anti-spam enforcement

Cooldowns are enforced via server-side timestamps.

### 4.1 Post cooldown

Helper: `postCooldownOk()`

- Reads caller’s user document:
  - `/users/{request.auth.uid}`
- If `lastPostAt` exists:
  - require `(request.time - lastPostAt) > 60s`
- If missing:
  - the write is allowed

### 4.2 Comment cooldown

Helper: `commentCooldownOk()`

- Reads caller’s user document:
  - `/users/{request.auth.uid}`
- If `lastCommentAt` exists:
  - require `(request.time - lastCommentAt) > 30s`
- If missing:
  - the write is allowed

### 4.3 Reputation cooldown

Helper: `reputationCooldownOk()`

- Uses `resource.data.lastReputationGain` (no extra reads)
- If present:
  - require `(request.time - lastReputationGain) > 300s`

This helper is used to gate reputation changes within the `/users/{userId}` update rule.

---

## 5) Collection-by-collection access matrix

### 5.1 `/users/{userId}`

**Read**
- Authenticated users can read their own profile
- Admin can read any profile

**Create**
- Only the caller can create their own user profile (`isOwner(userId)`)

**Update**
- Only the owner can update their profile
- Constraints:
  1. Reputation changes:
     - allowed only if `reputationCooldownOk()`
  2. Role updates:
     - if the write attempts to modify `role`, the new role must be exactly `farmer`

**Invariant enforced:** users cannot self-escalate into elevated roles.

---

### 5.2 `/feedback/{feedbackId}`

**Read / Delete**
- Admin only

**Create**
- Any authenticated user

---

### 5.3 `/posts/{postId}`

**Read**
- Any authenticated user

**Create** (must satisfy all)
- Authenticated
- `content` is valid (`postContentValid()`)
- `createdAt == request.time`
- `userId == request.auth.uid`
- `postCooldownOk()`

**Update**
- Authenticated AND either:
  - the update only affects `likes` or `commentsCount`, OR
  - caller is the owner AND the updated content passes `postContentValid()`

**Delete**
- Owner or admin

---

### 5.4 `/comments/{commentId}`

**Read**
- Authenticated

**Create** (must satisfy all)
- `text` validated (`commentContentValid()`)
- `createdAt == request.time`
- `userId == request.auth.uid`
- `commentCooldownOk()`

**Update**
- Authenticated AND either:
  - the update only affects `upvotes` or `downvotes`, OR
  - caller is the owner AND the updated comment text passes `commentContentValid()`

**Delete**
- Owner or admin

---

### 5.5 `/reports/{reportId}`

**Read / Write**
- Only `expert` or `admin`

---

### 5.6 `/marketplace/{itemId}`

**Read**
- Public read (`allow read: if true`)

**Write**
- `vendor` or `admin`

---

### 5.7 `/finance_applications/{applicationId}`

**Read**
- Authenticated AND either:
  - `resource.data.owner_uid == request.auth.uid`, OR
  - role is `admin` or `expert`

**Create**
- Authenticated AND `request.resource.data.owner_uid == request.auth.uid`

**Update**
- Authenticated AND either:
  - caller owns it, OR
  - role is `admin` or `expert`

**Delete**
- Admin only

---

### 5.8 `/notifications/{notificationId}`

**Read**
- Authenticated AND either:
  - `resource.data.userId == request.auth.uid`, OR
  - admin

**Write**
- Only `admin` or `system`

> Note: notifications are expected to be backend-generated (hence the `system` role).

---

### 5.9 `/supply_chain_batches/{batchId}` and `/nodes/{nodeId}`

Batch:
- **Read:** authenticated
- **Create/Update:** authenticated AND role in `['farmer','vendor','admin']`
- **Delete:** admin only

Node (nested under batch):
- **Read:** authenticated
- **Create/Update:** authenticated AND role in `['farmer','vendor','admin']`
- **Delete:** admin only

---

### 5.10 `/disaster_alerts/{alertId}`

**Read**
- Authenticated

**Create** (must satisfy all)
- Authenticated
- `reporterId == request.auth.uid`
- Valid enumerations and bounds:
  - `type` and `severity` constrained
  - `notes` type and max length (<= 100 chars)
  - `geohash` string; `lat` and `lng` numbers
  - `createdAt == request.time`

**Delete**
- Owner (`reporterId`) or admin

---

### 5.11 `/consultations/{consultationId}`

**Read**
- Authenticated AND either:
  - `resource.data.userId == request.auth.uid` OR
  - `resource.data.expertId == request.auth.uid` OR
  - admin

**Create**
- Authenticated
- `request.resource.data.userId == request.auth.uid`
- If `notes` exists, validate:
  - `notes` is a string
  - max length <= 1000
- `status` must be exactly `"scheduled"`

**Update**
- Authenticated AND either:
  - caller is the owning farmer (`resource.data.userId`), OR
  - caller is the assigned expert (`resource.data.expertId`), OR
  - admin
- Invariant: `request.resource.data.userId == resource.data.userId` (no user reassignment)

**Delete**
- Admin only

---

### 5.12 `/farm_documents/{docId}` (Farm Document Vault)

**Read**
- Authenticated owner: `resource.data.owner_uid == request.auth.uid`
- Admin: `hasRole('admin')`

**Create**
- Authenticated owner only: `request.resource.data.owner_uid == request.auth.uid`
- Required metadata fields and validation:
  - `owner_uid`: non-empty string
  - `title`: string, 1..200 chars
  - `fileName`: string, 1..255 chars
  - `contentType`: string, 1..100 chars
  - `storagePath`: string, 1..1024 chars
  - `createdAt` and `updatedAt`: must be server timestamps
  - Optional `tags`: array of strings, max 20 tags; each tag max 20 chars

**Update**
- Authenticated owner only (or admin)
- `owner_uid` is immutable: `request.resource.data.owner_uid == resource.data.owner_uid`
- Same metadata validation as create.

**Delete**
- Authenticated owner only (or admin)

---

### 5.13 `/_schema_migrations/{migrationId}`

**Read/Write**
- Admin only

---


## 6) Change management guidance

When evolving these rules:

1. Prefer tightening over loosening.
2. Preserve invariants:
   - consultation `userId` immutability
   - prevention of role self-escalation
   - reporter binding on disaster alerts
3. Keep validation local to rules.
   - If the frontend filters a field, the rules should still validate it.
4. Avoid adding expensive reads.
   - Extra `get()` calls inside rules can increase evaluation cost and latency.
5. Maintain token/claim consistency.
   - Do not introduce “default role” fallbacks.

---

## 7) Notes on performance and correctness

This ruleset intentionally uses helper functions and relies on:
- `resource.data` for update-time constraints whenever possible
- role claims in `request.auth.token`

This reduces unnecessary reads and helps ensure cooldown logic remains accurate.

