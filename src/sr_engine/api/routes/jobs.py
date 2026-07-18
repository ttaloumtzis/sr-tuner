from fastapi import APIRouter

from sr_engine.api.schemas import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=list[JobStatus])
async def list_jobs():
    from sr_engine.api.app import tasks

    records = tasks.list_jobs()
    return [
        JobStatus(
            job_id=r.job_id,
            job_type=r.job_type,
            status=r.status,
            created_at=r.created_at,
            started_at=r.started_at,
            completed_at=r.completed_at,
            error=r.error,
            result=r.result,
        )
        for r in records
    ]


@router.get("/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    from sr_engine.api.app import tasks

    rec = tasks.get_job(job_id)
    if not rec:
        from fastapi import HTTPException
        raise HTTPException(404, f"Job not found: {job_id}")
    return JobStatus(
        job_id=rec.job_id,
        job_type=rec.job_type,
        status=rec.status,
        created_at=rec.created_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        error=rec.error,
        result=rec.result,
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    from sr_engine.api.app import tasks

    ok = tasks.cancel_job(job_id)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(404, f"Job not found or already finished: {job_id}")
    return {"status": "cancelled", "job_id": job_id}