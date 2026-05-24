# Firestore ABAC Deployment Checklist

Use this checklist to ensure a smooth deployment of the attribute-based access control (ABAC) implementation.

## Pre-Deployment

### 1. Backup Current State
- [ ] Export current Firestore data
  ```bash
  gcloud firestore export gs://[BUCKET_NAME]/firestore-backup-$(date +%Y%m%d)
  ```
- [ ] Save current Firestore rules
  ```bash
  firebase firestore:rules > firestore.rules.backup
  ```
- [ ] Document current user count and organization structure
- [ ] Take screenshots of current Firebase Console state

### 2. Review Documentation
- [ ] Read [FIRESTORE_ABAC_SECURITY.md](docs/FIRESTORE_ABAC_SECURITY.md)
- [ ] Read [ABAC_QUICK_START.md](docs/ABAC_QUICK_START.md)
- [ ] Review [ABAC_ARCHITECTURE.md](docs/ABAC_ARCHITECTURE.md)
- [ ] Understand rollback procedures

### 3. Environment Preparation
- [ ] Verify Firebase Admin SDK is configured
  ```bash
  python -c "import firebase_admin; print('✓ SDK available')"
  ```
- [ ] Verify Firebase CLI is installed and authenticated
  ```bash
  firebase --version
  firebase projects:list
  ```
- [ ] Set up staging/test environment (if available)
- [ ] Ensure GOOGLE_APPLICATION_CREDENTIALS is set
  ```bash
  echo $GOOGLE_APPLICATION_CREDENTIALS
  ```

### 4. Code Review
- [ ] Review changes to `firestore.rules`
- [ ] Review `auth_claims.py` module
- [ ] Review `frontend/lib/firestore-abac.js` helpers
- [ ] Review `migration_add_org_id.py` script
- [ ] Run linters and code quality checks

---

## Deployment Phase 1: Backend Setup

### 5. Install Dependencies
- [ ] Install Python dependencies
  ```bash
  pip install firebase-admin
  ```
- [ ] Verify imports work
  ```bash
  python -c "from auth_claims import CustomClaimsManager; print('✓ Module loaded')"
  ```

### 6. Test Custom Claims Module
- [ ] Test setting claims on a test user
  ```python
  from auth_claims import set_user_claims
  set_user_claims('test_user_uid', 'farmer', 'test_org')
  ```
- [ ] Verify claims are set correctly
  ```python
  from auth_claims import get_user_claims
  claims = get_user_claims('test_user_uid')
  print(claims)  # Should show role and organizationId
  ```
- [ ] Test updating role
- [ ] Test updating organization

---

## Deployment Phase 2: Data Migration

### 7. Prepare Migration
- [ ] Identify all collections that need organizationId
- [ ] Determine default organization ID for orphaned data
- [ ] Estimate migration time (count documents)
  ```python
  # In Python console
  from firebase_admin import firestore
  db = firestore.client()
  count = len(list(db.collection('posts').stream()))
  print(f"Posts to migrate: {count}")
  ```

### 8. Run Migration (Dry Run)
- [ ] Run migration in dry-run mode
  ```bash
  python migration_add_org_id.py --dry-run
  ```
- [ ] Review dry-run output
- [ ] Verify no errors in dry-run
- [ ] Check that organizationId assignments look correct

### 9. Run Migration (Live)
- [ ] Run migration for real
  ```bash
  python migration_add_org_id.py --default-org legacy_org_001
  ```
- [ ] Monitor migration progress
- [ ] Check for errors in output
- [ ] Verify migration statistics

### 10. Verify Migration
- [ ] Check Firebase Console > Firestore
- [ ] Verify documents have organizationId field
- [ ] Spot-check a few documents manually
- [ ] Verify no documents were lost
- [ ] Check that migratedAt timestamp is present

---

## Deployment Phase 3: Firestore Configuration

### 11. Deploy Indexes
- [ ] Review `firestore.indexes.json`
- [ ] Deploy indexes
  ```bash
  firebase deploy --only firestore:indexes
  ```
- [ ] Wait for indexes to build (check Firebase Console)
- [ ] Verify all indexes show "Enabled" status
- [ ] Test queries with new indexes

**⏰ Index Build Time**: Can take 30-60 minutes for large datasets

### 12. Test Rules (Optional - Staging)
- [ ] Deploy rules to staging environment first
  ```bash
  firebase use staging
  firebase deploy --only firestore:rules
  ```
- [ ] Test with staging data
- [ ] Verify organization isolation works
- [ ] Test admin access
- [ ] Test role-based permissions

### 13. Deploy Rules (Production)
- [ ] Switch to production project
  ```bash
  firebase use production
  ```
- [ ] Deploy Firestore rules
  ```bash
  firebase deploy --only firestore:rules
  ```
- [ ] Verify deployment success
- [ ] Check Firebase Console > Firestore > Rules tab
- [ ] Review rules for syntax errors

---

## Deployment Phase 4: Frontend Updates

### 14. Update Frontend Code
- [ ] Add `frontend/lib/firestore-abac.js` to project
- [ ] Initialize ABAC in main App component
  ```javascript
  import { initABAC } from './lib/firestore-abac';
  useEffect(() => { initABAC(); }, []);
  ```
- [ ] Update document creation to use `createDocWithOrg()`
- [ ] Update queries to use `queryOrgDocs()`
- [ ] Update document updates to use `updateDocSafe()`

### 15. Update User Registration
- [ ] Modify registration endpoint to set custom claims
  ```python
  from auth_claims import set_user_claims
  set_user_claims(user.uid, 'farmer', org_id)
  ```
- [ ] Add organization assignment logic (invite codes, etc.)
- [ ] Test registration flow end-to-end
- [ ] Verify new users get custom claims

### 16. Build and Deploy Frontend
- [ ] Run frontend build
  ```bash
  cd frontend
  npm run build
  ```
- [ ] Test build locally
  ```bash
  npm run preview
  ```
- [ ] Deploy to production
- [ ] Verify deployment success

---

## Deployment Phase 5: Testing

### 17. Functional Testing
- [ ] Create test users in different organizations
- [ ] Test user A cannot see user B's data
- [ ] Test user A can see their own organization's data
- [ ] Test admin can see all organizations' data
- [ ] Test creating documents (should include organizationId)
- [ ] Test updating documents (organizationId should be immutable)
- [ ] Test deleting documents

### 18. Security Testing
- [ ] Attempt to read data from other organization (should fail)
- [ ] Attempt to change organizationId on update (should fail)
- [ ] Attempt to escalate role (should fail)
- [ ] Test with invalid/missing organizationId (should fail)
- [ ] Test with expired auth token (should fail)

### 19. Performance Testing
- [ ] Measure query performance with new indexes
- [ ] Check for slow queries in Firebase Console
- [ ] Verify caching is working (check browser console)
- [ ] Monitor Firestore read/write counts

### 20. Run Automated Tests
- [ ] Run backend tests
  ```bash
  pytest tests/test_firestore_security.py -v
  ```
- [ ] Run frontend tests (if available)
- [ ] Verify all tests pass

---

## Deployment Phase 6: Monitoring

### 21. Enable Monitoring
- [ ] Enable Firestore audit logging
- [ ] Set up Cloud Logging queries
  ```
  resource.type="firestore_database"
  severity="ERROR"
  ```
- [ ] Set up alerts for permission denied errors
- [ ] Monitor error rates in Firebase Console

### 22. Monitor Initial Traffic
- [ ] Watch for permission denied errors
- [ ] Check for missing organizationId errors
- [ ] Monitor query performance
- [ ] Check for unexpected behavior
- [ ] Review user feedback/support tickets

### 23. Verify Data Integrity
- [ ] Spot-check documents in Firebase Console
- [ ] Verify organizationId is present on all new documents
- [ ] Check that no data was lost during migration
- [ ] Verify user counts match pre-migration

---

## Post-Deployment

### 24. Documentation Updates
- [ ] Update README with ABAC information
- [ ] Document organization assignment process
- [ ] Create admin guide for managing organizations
- [ ] Update API documentation
- [ ] Share deployment summary with team

### 25. Team Training
- [ ] Train developers on new ABAC helpers
- [ ] Train support team on organization concepts
- [ ] Document common issues and solutions
- [ ] Create troubleshooting guide

### 26. Cleanup
- [ ] Remove old backup files (after verification period)
- [ ] Archive migration logs
- [ ] Update project documentation
- [ ] Close related tickets/issues

---

## Rollback Procedures

### If Issues Arise

#### Option 1: Revert Firestore Rules
```bash
firebase deploy --only firestore:rules --version <previous-version>
```

#### Option 2: Disable ABAC Temporarily
Edit `firestore.rules` and add at the top:
```javascript
function isABACEnabled() {
  return false;  // Temporarily disable
}
```

Then update all rules to check:
```javascript
allow read: if !isABACEnabled() || canAccessOrg(...);
```

#### Option 3: Restore from Backup
```bash
gcloud firestore import gs://[BUCKET_NAME]/firestore-backup-[DATE]
```

---

## Success Criteria

Deployment is successful when:

- [ ] ✅ All indexes are built and enabled
- [ ] ✅ Firestore rules are deployed without errors
- [ ] ✅ All documents have organizationId field
- [ ] ✅ Users can only access their organization's data
- [ ] ✅ Admins can access all organizations' data
- [ ] ✅ No permission denied errors for legitimate requests
- [ ] ✅ Performance is acceptable (no significant slowdown)
- [ ] ✅ All automated tests pass
- [ ] ✅ No critical bugs reported
- [ ] ✅ Monitoring shows normal operation

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Pre-Deployment | 1-2 hours | Review, backup, preparation |
| Backend Setup | 30 minutes | Install dependencies, test |
| Data Migration | 30-60 minutes | Depends on data volume |
| Firestore Config | 1-2 hours | Includes index build time |
| Frontend Updates | 1-2 hours | Code changes, testing |
| Testing | 1-2 hours | Comprehensive testing |
| Monitoring | Ongoing | First 24-48 hours critical |
| **Total** | **6-10 hours** | Spread over 1-2 days |

---

## Emergency Contacts

- **Firebase Admin**: [Name/Email]
- **Backend Lead**: [Name/Email]
- **Frontend Lead**: [Name/Email]
- **DevOps**: [Name/Email]
- **On-Call**: [Phone/Slack]

---

## Notes

Use this space for deployment-specific notes:

```
Date: _______________
Deployed by: _______________
Environment: _______________
Issues encountered: _______________
Resolution: _______________
```

---

## Sign-Off

- [ ] Backend Lead: _________________ Date: _______
- [ ] Frontend Lead: _________________ Date: _______
- [ ] QA Lead: _________________ Date: _______
- [ ] Product Owner: _________________ Date: _______

---

**Last Updated**: 2026-05-13  
**Version**: 1.0  
**Status**: Ready for use
