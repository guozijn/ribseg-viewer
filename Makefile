.PHONY: install dev

install:
	.venv/bin/pip install -e "backend[dev]"
	cd frontend && npm install

dev:
	@trap 'kill 0' EXIT; \
	(cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8110) & \
	(cd frontend && npm run dev -- --port 5174)
