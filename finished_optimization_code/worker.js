// Cloudflare Worker — Greenhouse API
// Bindings: GREENHOUSE_KV (KV namespace), PASSWORD (secret), DB (D1 database)

export default {
  async fetch(request, env, ctx) {
    const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
      "Content-Type": "application/json"
    };

    function applyPumpUpdate(data) {
      if (typeof data.pump === "number" && !isNaN(data.pump)) {
        data.pump = Math.max(0, Math.min(14, data.pump));
        data.pumpTimestamp = Date.now();
      } else {
        delete data.pump;
      }
    }

    // Non-blocking D1 append — runs after response is sent
    function logObservation(state, ts) {
      if (!env.DB) return;
      const soil6 = (state.soil || []).slice(0, 6);
      while (soil6.length < 6) soil6.push(null);
      ctx.waitUntil(
        env.DB.prepare(
          `INSERT INTO observations
           (timestamp_ms, soc, temperature, humidity,
            soil1, soil2, soil3, soil4, soil5, soil6,
            battery_in_v, battery_in_a, battery_out_v, battery_out_a)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
          .bind(
            ts,
            state.soc ?? null,
            state.temperature ?? null,
            state.humidity ?? null,
            ...soil6,
            state.battery?.in?.voltage ?? null,
            state.battery?.in?.amps ?? null,
            state.battery?.out?.voltage ?? null,
            state.battery?.out?.amps ?? null
          )
          .run()
          .catch((e) => console.error("D1 obs log failed:", e))
      );
    }

    function logAction(ts, lights, vent, pumpMinutes) {
      if (!env.DB) return;
      ctx.waitUntil(
        env.DB.prepare(
          `INSERT INTO actions (timestamp_ms, lights, vent, pump_minutes)
           VALUES (?, ?, ?, ?)`
        )
          .bind(ts, lights ? 1 : 0, vent ? 1 : 0, pumpMinutes ?? 0)
          .run()
          .catch((e) => console.error("D1 action log failed:", e))
      );
    }

    function logSystem(ts, mpc) {
      if (!env.DB || !mpc) return;
      ctx.waitUntil(
        env.DB.prepare(
          `INSERT INTO system_log
           (timestamp_ms, mpc_solve_time_ms, mpc_status, mpc_soc_end,
            mpc_temp_end, mpc_pump_total_min, mpc_lights_total_h, mpc_cost,
            pred_soc_1step, pred_temp_1step_c, pred_soil_1step,
            pv_predicted_w, pv_actual_w, pv_daily_kwh, cloud_cover)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
          .bind(
            ts,
            mpc.solve_time_ms ?? null,
            mpc.status ?? null,
            mpc.soc_end ?? null,
            mpc.temp_end ?? null,
            mpc.pump_total_min ?? null,
            mpc.lights_total_h ?? null,
            mpc.cost ?? null,
            mpc.pred_soc_1step ?? null,
            mpc.pred_temp_1step_c ?? null,
            mpc.pred_soil_1step ?? null,
            mpc.pv_predicted_w ?? null,
            mpc.pv_actual_w ?? null,
            mpc.pv_daily_kwh ?? null,
            mpc.cloud_cover ?? null
          )
          .run()
          .catch((e) => console.error("D1 system log failed:", e))
      );
    }

    try {
      const CLOUDINARY_CLOUD_NAME = "dde5usvck";
      const CLOUDINARY_UPLOAD_PRESET = "p715jmvb";

      if (request.method === "OPTIONS") {
        return new Response(null, { headers });
      }

      const defaultState = {
        lights: false,
        vent: false,
        pump: 0,
        pumpTimestamp: null,
        takePicture: false,
        soil: [null, null, null, null, null, null],
        temperature: null,
        humidity: null,
        picture: null,
        pictureTimestamp: null,
        battery: {
          in: { voltage: null, amps: null },
          out: { voltage: null, amps: null }
        },
        soc: null,
        plantState: "GROWING"
      };

      // ── GET ──────────────────────────────────────────────────────────
      if (request.method === "GET") {
        const parsed = new URL(request.url);

        // GET /api/history?hours=6  — historical data for dashboard graphs
        if (parsed.pathname === "/api/history") {
          if (!env.DB) {
            return new Response(
              JSON.stringify({ observations: [], actions: [], error: "No D1 database" }),
              { headers }
            );
          }

          const hours = Math.min(parseFloat(parsed.searchParams.get("hours") || "24"), 168);
          const since = Date.now() - hours * 3600 * 1000;

          const [obsResult, actResult, sysResult] = await env.DB.batch([
            env.DB.prepare(
              `SELECT * FROM observations WHERE timestamp_ms >= ? ORDER BY timestamp_ms ASC`
            ).bind(since),
            env.DB.prepare(
              `SELECT * FROM actions WHERE timestamp_ms >= ? ORDER BY timestamp_ms ASC`
            ).bind(since),
            env.DB.prepare(
              `SELECT * FROM system_log WHERE timestamp_ms >= ? ORDER BY timestamp_ms ASC`
            ).bind(since),
          ]);

          return new Response(
            JSON.stringify({
              observations: obsResult.results || [],
              actions: actResult.results || [],
              system_log: sysResult.results || [],
            }),
            { headers }
          );
        }

        // GET / — current state from KV
        const stored = await env.GREENHOUSE_KV.get("state", "json");
        const state = stored || defaultState;
        return new Response(JSON.stringify(state), { headers });
      }

      // ── POST ─────────────────────────────────────────────────────────
      if (request.method === "POST") {
        const data = await request.json();

        if (data.password !== env.PASSWORD) {
          return new Response(
            JSON.stringify({ error: "Wrong password" }),
            { status: 403, headers }
          );
        }
        delete data.password;

        applyPumpUpdate(data);

        const stored = await env.GREENHOUSE_KV.get("state", "json");
        const currentState = stored || defaultState;
        const newState = { ...currentState, ...data };

        await env.GREENHOUSE_KV.put("state", JSON.stringify(newState));

        // Log command to D1
        const ts = Date.now();
        logAction(ts, newState.lights, newState.vent, newState.pump);

        return new Response(JSON.stringify(newState), { headers });
      }

      // ── PUT ──────────────────────────────────────────────────────────
      if (request.method === "PUT") {
        const data = await request.json();

        if (data.password !== env.PASSWORD) {
          return new Response(
            JSON.stringify({ error: "Wrong password" }),
            { status: 403, headers }
          );
        }
        delete data.password;

        applyPumpUpdate(data);

        const stored = await env.GREENHOUSE_KV.get("state", "json");
        const currentState = stored || defaultState;

        if (Array.isArray(data.soil)) {
          const soil = Array.from({ length: 6 }, (_, i) => {
            const cs = currentState.soil;
            return (cs && cs[i] != null) ? cs[i] : null;
          });
          for (let i = 0; i < 6; i++) {
            if (data.soil[i] !== undefined) soil[i] = data.soil[i];
          }
          data.soil = soil;
        }

        // Upload base64 picture to Cloudinary
        if (typeof data.picture === "string" && data.picture.startsWith("data:")) {
          try {
            const uploadForm = new FormData();
            uploadForm.append("file", data.picture);
            uploadForm.append("upload_preset", CLOUDINARY_UPLOAD_PRESET);

            const uploadResponse = await fetch(
              `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/image/upload`,
              { method: "POST", body: uploadForm }
            );
            const uploadResult = await uploadResponse.json();
            if (uploadResult.secure_url) {
              data.picture = uploadResult.secure_url;
            } else {
              console.error("Cloudinary upload failed:", uploadResult);
              delete data.picture;
            }
          } catch (e) {
            console.error("Cloudinary exception:", e);
            delete data.picture;
          }
        }

        if (typeof data.picture === "string") {
          data.pictureTimestamp = Date.now();
        }

        if (typeof data.temperature !== "number") delete data.temperature;
        if (typeof data.humidity !== "number") delete data.humidity;

        if (typeof data.soc !== "number" || isNaN(data.soc)) {
          delete data.soc;
        } else {
          data.soc = Math.max(0, Math.min(100, data.soc));
        }

        if (data.battery) {
          const cb = currentState.battery || defaultState.battery;
          data.battery = {
            in: { ...cb.in, ...(data.battery.in || {}) },
            out: { ...cb.out, ...(data.battery.out || {}) }
          };
        }

        const newState = { ...currentState, ...data };
        await env.GREENHOUSE_KV.put("state", JSON.stringify(newState));

        // Log observation to D1
        const ts = Date.now();
        logObservation(newState, ts);

        // Also log action if commands were updated
        if (data.lights !== undefined || data.vent !== undefined || data.pump !== undefined) {
          logAction(ts, newState.lights, newState.vent, newState.pump);
        }

        // Log MPC result if provided
        if (data.mpc) {
          logSystem(ts, data.mpc);
        }

        return new Response(JSON.stringify(newState), { headers });
      }

      return new Response(
        JSON.stringify({ error: "Method Not Allowed" }),
        { status: 405, headers }
      );

    } catch (err) {
      console.error("WORKER ERROR:", err);
      return new Response(
        JSON.stringify({ success: false, error: err.message, stack: err.stack }),
        { status: 500, headers }
      );
    }
  }
};
