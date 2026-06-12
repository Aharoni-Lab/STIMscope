# Operational targets.
#
# `make fresh` is the canonical way to restart the GUI. `docker-compose restart`
# is intentionally NOT exposed here — it preserves USB camera handles and
# leaves ZMQ ports 5558/5560 bound to dead PIDs. Always down+up..PHONY: build fresh up down logs logs-tail logs-summary logs-stop-tail \
        status shell gui-shell rebuild-projector demo demo-preview demo-verify \
        bandit bandit-low pip-audit test help

help:
	@echo "STIMscope / CRISPI targets:"
	@echo "  make build             - Build crispi:latest image (auto-detects JetPack)"
	@echo "  make fresh             - down + up -d gui: restart the GUI"
	@echo "  make up                - same as 'make fresh' but no down first"
	@echo "  make down              - take all services down"
	@echo "  make logs              - tail GUI logs"
	@echo "  make status            - show container + image build info"
	@echo "  make shell             - open bash inside a one-off crispi:latest container"
	@echo "  make gui-shell         - open bash inside the running gui container"
	@echo "  make rebuild-projector - recompile the C++ projector binary on the host"
	@echo "  make demo              - record the DMD demo (camera) + auto-verify"
	@echo "  make demo-preview      - projection-only smoke (no camera, quick)"
	@echo "  make demo-verify B=dir - re-run the sync/accuracy report on a bundle"

build:./build.sh

fresh:
	@echo ">>> make fresh: stopping old containers, then launching GUI"
	-docker stop crispi-gui 2>/dev/null
	-docker rm crispi-gui 2>/dev/null
	@# X11 setup: older `xhost +local:docker` silently no-ops
	@# when DISPLAY is unset in make's shell. GDM 3.x stores the X auth
	@# under /run/user/$$UID/gdm/Xauthority, NOT ~/.Xauthority. Authorize
	@# the container's UID (root) against the running X server + bind-mount
	@# a readable copy of the Xauthority cookie so Qt can authenticate.
	@DISPLAY=$${DISPLAY:-:0} XAUTHORITY=/run/user/$$(id -u)/gdm/Xauthority \
		xhost +SI:localuser:root 2>/dev/null || true
	@if [ -f /run/user/$$(id -u)/gdm/Xauthority ]; then \
		cp /run/user/$$(id -u)/gdm/Xauthority /tmp/docker.xauth && chmod 644 /tmp/docker.xauth; \
	fi
	docker run --rm -d \
		--name crispi-gui \
		--runtime nvidia \
		--privileged \
		--network host \
		-e DISPLAY=$${DISPLAY:-:0} \
		-e XAUTHORITY=/tmp/docker.xauth \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-e QT_X11_NO_MITSHM=1 \
		-e PYTHONUNBUFFERED=1 \
		-e GENICAM_GENTL64_PATH=/opt/ids-peak/lib/aarch64-linux-gnu/ids-peak/cti \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /tmp/docker.xauth:/tmp/docker.xauth:ro \
		-v $(CURDIR)/STIMscope/STIMViewer_CRISPI:/app/STIMViewer_CRISPI \
		-v $(CURDIR)/STIMscope/ZMQ_sender_mask:/app/ZMQ_sender_mask \
		-v $(CURDIR)/data:/data \
		-v $${HOME}:/host_home:ro \
		-v /media:/host_media:ro \
		-v $${IDS_PEAK_PATH:-/opt/ids-peak}:/opt/ids-peak:ro \
		--device /dev/bus/usb:/dev/bus/usb \
		--device /dev/gpiochip1:/dev/gpiochip1 \
		crispi:latest \
		/app/STIMViewer_CRISPI/main_gui.pyw
	@echo ">>> GUI running as 'crispi-gui'. Use 'make logs' to follow."

up: fresh

down:
	-docker stop crispi-gui 2>/dev/null
	-docker rm crispi-gui 2>/dev/null

logs:
	docker logs -f crispi-gui

# Durable, on-disk, append-only capture pattern:
# durable, on-disk, append-only capture of the crispi-gui container log.
# Runs in background so the operator can keep using the shell. The log file
# stays after the container exits — useful for forensic re-analysis with
# grep/awk/jq after a session.
#
#   make logs-tail              # start background tail; print log path
#   make logs-summary           # grep the latest log for milestone events
#   make logs-stop-tail         # kill the background tail
#
# Logs live at /tmp/crispi-<TS>.log — durable until /tmp is cleared. Symlink
# /tmp/crispi-latest.log always points at the most recent capture.

logs-tail:
	@TS=$$(date +%Y%m%d_%H%M%S); \
	LOG=/tmp/crispi-$$TS.log; \
	if pgrep -f 'docker logs -f crispi-gui' >/dev/null 2>&1; then \
		echo ">>> A logs-tail is already running:"; \
		pgrep -af 'docker logs -f crispi-gui'; \
		echo "    Stop it first with 'make logs-stop-tail' if you want a fresh capture."; \
		exit 0; \
	fi; \
	nohup docker logs -f crispi-gui > $$LOG 2>&1 & \
	disown; \
	ln -sf $$LOG /tmp/crispi-latest.log; \
	echo ">>> Background tail started: $$LOG"; \
	echo "    Symlinked at /tmp/crispi-latest.log"; \
	echo "    Inspect with:  tail -200 /tmp/crispi-latest.log"; \
	echo "                   grep -E 'STREAMER|MASK|finalized|Traceback' /tmp/crispi-latest.log"; \
	echo "    Stop with:     make logs-stop-tail"

logs-summary:
	@if [ ! -e /tmp/crispi-latest.log ]; then \
		echo "No log capture running. Start with 'make logs-tail' after 'make fresh'."; \
		exit 1; \
	fi
	@echo "=== Latest capture: $$(readlink -f /tmp/crispi-latest.log) ==="
	@echo "=== Line count + size ==="
	@wc -l /tmp/crispi-latest.log; du -h /tmp/crispi-latest.log | cut -f1 | xargs -I{} echo "size: {}"
	@echo "=== Last Recording finalize ==="
	@grep "Recording finalized" /tmp/crispi-latest.log | tail -1 || echo "(none yet)"
	@echo "=== Last STREAMER summary ==="
	@grep "\[STREAMER\] stopped" /tmp/crispi-latest.log | tail -1 || echo "(none yet)"
	@echo "=== Trial milestones (last 5) ==="
	@grep -E "\[MASK\] k=|\[TRACE-DBG\] k=" /tmp/crispi-latest.log | tail -5 || echo "(none yet)"
	@echo "=== Any errors / tracebacks (last 5) ==="
	@grep -E "Traceback|^E[A-Z]|FAILED|SIGSEGV|RuntimeError|ValueError|Critical" /tmp/crispi-latest.log | tail -5 || echo "(none — clean)"

logs-stop-tail:
	@if pgrep -f 'docker logs -f crispi-gui' >/dev/null 2>&1; then \
		pkill -f 'docker logs -f crispi-gui' && echo ">>> Stopped background tail."; \
	else \
		echo "(no logs-tail process running)"; \
	fi

status:
	@echo "=== image ==="
	@docker images crispi:latest --format '{{.ID}} {{.CreatedAt}} {{.Size}}' || echo "image not built"
	@echo "=== build_info.txt (from image) ==="
	@docker run --rm --entrypoint cat crispi:latest /app/build_info.txt 2>/dev/null || echo "(not present — rebuild with current Dockerfile)"
	@echo "=== containers ==="
	@docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Command}}' | grep -E 'crispi|gui' || echo "nothing running"

shell:
	docker run --rm -it \
		--runtime nvidia --privileged --network host \
		-v $(CURDIR)/STIMscope/STIMViewer_CRISPI:/app/STIMViewer_CRISPI \
		-v $(CURDIR)/data:/data \
		-v $${IDS_PEAK_PATH:-/opt/ids-peak}:/opt/ids-peak:ro \
		crispi:latest bash

gui-shell:
	docker exec -it crispi-gui bash

# Rebuild the projector binary directly on the host. The crispi service's live
# mount of ZMQ_sender_mask means the new binary is picked up on the next
# `make fresh` without a full image rebuild.
rebuild-projector:
	cd STIMscope/ZMQ_sender_mask && \
	g++ -O2 -std=c++17 main.cpp -o projector -lglfw -lGL -lzmq -lgpiod -lpthread -lGLEW
	@echo ">>> rebuilt projector; run 'make fresh' to pick it up"

# ── DMD demo recorder ─────────────────────────────────────────────────────────
# `make demo`         — full hardware capture (boot + projector + camera) of the
#                       deterministic mask sequence, then auto-verify (sync +
#                       accuracy PASS/FAIL + synced_frames.csv). Args pass via
#                       ARGS=, e.g.  make demo ARGS="--sequence density"
#                       Output dir overridable:  make demo OUT_DIR=/mnt/ssd/demo
# `make demo-preview` — projection-only smoke (no camera, fast) to eyeball it.
# `make demo-verify`  — re-run the report on an existing bundle: make demo-verify B=<dir>
# `make demo-compose` — build the RAW|PROJECTION|CAMERA triptych TIFF for a
#                       bundle: make demo-compose B=<dir> [ARGS="--all"]
demo:./scripts/run_demo.sh $(ARGS)

demo-preview:./scripts/run_demo.sh --no-camera --hold-scale 0.4 $(ARGS)

demo-verify:
	@test -n "$(B)" || { echo "usage: make demo-verify B=<bundle-dir>"; exit 2; }
	python3 tools/demo/verify.py --bundle-dir "$(B)"

demo-compose:
	@test -n "$(B)" || { echo "usage: make demo-compose B=<bundle-dir> [ARGS=\"--all\"]"; exit 2; }
	python3 tools/demo/composer.py --bundle-dir "$(B)" --all \
		--out "$(B)/demo_composite.tiff" $(ARGS)

# ── Quality / security gates ──────────────────────────────────────────────────
#
# `make bandit`     — medium+ severity scan (gate intended to remain clean)
# `make bandit-low` — full low-severity report (advisory; many try/except:pass
#                     findings are surfaced for triage, not blocking)
# `make pip-audit`  — installed-deps CVE scan
# `make test`       — full pytest run inside the image

bandit:
	docker run --rm --entrypoint bash \
		-v $(CURDIR):/repo:rw -w /repo crispi:latest \
		-c "export PATH=/opt/conda/bin:\$$PATH && pip install -q bandit && \
		    bandit -r STIMscope/STIMViewer_CRISPI/ \
		      --exclude '*/tests/*,*/legacy/*' \
		      -ll"

bandit-low:
	docker run --rm --entrypoint bash \
		-v $(CURDIR):/repo:rw -w /repo crispi:latest \
		-c "export PATH=/opt/conda/bin:\$$PATH && pip install -q bandit && \
		    bandit -r STIMscope/STIMViewer_CRISPI/ \
		      --exclude '*/tests/*,*/legacy/*' \
		      -l"

pip-audit:
	docker run --rm --entrypoint bash \
		-v $(CURDIR):/repo:rw -w /repo crispi:latest \
		-c "export PATH=/opt/conda/bin:\$$PATH && pip install -q pip-audit && pip-audit"

test:
	docker run --rm --runtime=nvidia --entrypoint bash \
		-v $(CURDIR):/repo:rw -w /repo crispi:latest \
		-c "export PATH=/opt/conda/bin:\$$PATH && \
		    pip install -q -r requirements-dev.txt scikit-learn && \
		    pytest -q"

# ── Wiki preview (local Gollum, port 4567) ─────────────────────────────────────
#
# `make wiki-preview`       — start local Gollum container against wiki/ folder
# `make wiki-preview-stop`  — tear down the preview container
# `make wiki-preview-refresh` — pick up edits without restarting
#
# Browse: http://localhost:4567  (or http://<jetson-IP>:4567)
# Same engine GitHub itself uses, so layout/links/sidebar match the real wiki.

WIKI_PREVIEW_DIR := /tmp/crispi-wiki-preview

wiki-preview:
	@rm -rf $(WIKI_PREVIEW_DIR)
	@mkdir -p $(WIKI_PREVIEW_DIR)
	@cp wiki/*.md $(WIKI_PREVIEW_DIR)/
	@cd $(WIKI_PREVIEW_DIR) && git init -q && git checkout -b main -q 2>/dev/null && \
	  git add. && git -c user.email=preview@local -c user.name=preview commit -q -m "snapshot"
	@docker rm -f crispi-wiki-preview >/dev/null 2>&1 || true
	@docker run -d --name crispi-wiki-preview -p 4567:4567 \
	  -v $(WIKI_PREVIEW_DIR):/wiki gollumwiki/gollum >/dev/null
	@sleep 3
	@echo ""
	@echo "✅ Wiki preview running at:"
	@echo "   http://localhost:4567/Home"
	@echo "   (or http://<jetson-IP>:4567/Home from another machine)"
	@echo ""
	@echo "After editing wiki/*.md:  make wiki-preview-refresh"
	@echo "When done:                make wiki-preview-stop"

wiki-preview-refresh:
	@cp wiki/*.md $(WIKI_PREVIEW_DIR)/
	@cd $(WIKI_PREVIEW_DIR) && git add. && \
	  git -c user.email=preview@local -c user.name=preview commit -q -m "refresh" 2>/dev/null || true
	@docker restart crispi-wiki-preview >/dev/null
	@echo "✅ Refreshed: http://localhost:4567/Home"

wiki-preview-stop:
	@docker rm -f crispi-wiki-preview >/dev/null 2>&1 || true
	@rm -rf $(WIKI_PREVIEW_DIR)
	@echo "✅ Wiki preview stopped + cleaned up"
