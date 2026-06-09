@router.post("/shadow/start")
async def start_shadow_evaluation(
    request: Request,
    production_model: str,
    candidate_model: str,
):
    """Start a shadow evaluation. Requires admin or expert role."""

    user_id = await _require_admin_auth(request)

    if shadow_evaluator is None:
        raise HTTPException(
            status_code=500,
            detail="Not initialized",
        )

    logger.info(
        "governance.shadow_evaluation.request "
        "user_id=%s production_model=%s candidate_model=%s",
        user_id,
        production_model,
        candidate_model,
    )

    try:
        eval_id = shadow_evaluator.start_shadow_evaluation(
            production_model,
            candidate_model,
        )

        logger.info(
            "governance.shadow_evaluation.success "
            "user_id=%s eval_id=%s production_model=%s candidate_model=%s",
            user_id,
            eval_id,
            production_model,
            candidate_model,
        )

        return {
            "success": True,
            "eval_id": eval_id,
        }

    except Exception:
        logger.exception(
            "governance.shadow_evaluation.failed "
            "user_id=%s production_model=%s candidate_model=%s",
            user_id,
            production_model,
            candidate_model,
        )
        raise


@router.post("/versions/register")
async def register_model_version(
    request: Request,
    data: RegisterModelVersionRequest,
):
    """Register a new model version. Requires admin or expert role."""

    user_id = await _require_admin_auth(request)

    if version_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Not initialized",
        )

    logger.info(
        "governance.register_version.request "
        "user_id=%s model_name=%s model_path=%s rmse=%s r2_score=%s",
        user_id,
        data.model_name,
        data.model_path,
        data.rmse,
        data.r2_score,
    )

    try:
        version_id = version_manager.register_version(
            data.model_name,
            data.model_path,
            data.rmse,
            data.r2_score,
            data.metadata,
        )

        logger.info(
            "governance.register_version.success "
            "user_id=%s model_name=%s version_id=%s",
            user_id,
            data.model_name,
            version_id,
        )

        return {
            "success": True,
            "version_id": version_id,
        }

    except Exception:
        logger.exception(
            "governance.register_version.failed "
            "user_id=%s model_name=%s",
            user_id,
            data.model_name,
        )
        raise


@router.post("/versions/promote")
async def promote_model_version(
    request: Request,
    version_id: str,
):
    """Promote a model version to production. Requires admin role."""

    user_id = await _require_admin_auth(request)

    if version_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Not initialized",
        )

    logger.info(
        "governance.promote_version.request "
        "user_id=%s version_id=%s",
        user_id,
        version_id,
    )

    try:
        version_manager.promote_version(version_id)

        production_version = version_manager.get_production_version()

        logger.info(
            "governance.promote_version.success "
            "user_id=%s version_id=%s production_version=%s",
            user_id,
            version_id,
            production_version,
        )

        return {
            "success": True,
            "production_version": production_version,
        }

    except Exception:
        logger.exception(
            "governance.promote_version.failed "
            "user_id=%s version_id=%s",
            user_id,
            version_id,
        )
        raise


@router.post("/versions/rollback")
async def rollback_model_version(
    request: Request,
    version_id: str,
):
    """Roll back to a previous model version. Requires admin role."""

    user_id = await _require_admin_auth(request)

    if version_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Not initialized",
        )

    logger.info(
        "governance.rollback_version.request "
        "user_id=%s version_id=%s",
        user_id,
        version_id,
    )

    try:
        version_manager.rollback_to_version(version_id)

        production_version = version_manager.get_production_version()

        logger.info(
            "governance.rollback_version.success "
            "user_id=%s target_version=%s production_version=%s",
            user_id,
            version_id,
            production_version,
        )

        return {
            "success": True,
            "production_version": production_version,
        }

    except Exception:
        logger.exception(
            "governance.rollback_version.failed "
            "user_id=%s version_id=%s",
            user_id,
            version_id,
        )
        raise
