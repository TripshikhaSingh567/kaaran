"""
Lightweight FastAPI Backend for KAARAN on Render.
Exposes REST endpoints for simulation rollouts, SCM predictions, and artifact serving.
"""

import os
import sys
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.sim import HeatedTankSimulator
from python.models import EnsembleModel

app = FastAPI(title="KAARAN Causal World Model Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SCM ensemble if weights exist
scm_model = None
sim = HeatedTankSimulator()

weights_path = os.path.join("models", "weights_scm.json")
if not os.path.exists(weights_path):
    weights_path = os.path.join("web", "public", "artifacts", "weights_scm.json")

if os.path.exists(weights_path):
    try:
        with open(weights_path, "r") as f:
            scm_dict = json.load(f)
        scm_model = EnsembleModel(scm_dict["in_dim"], scm_dict["num_members"], scm_dict["hidden_dim"])
        scm_model.load_from_dict(scm_dict)
    except Exception as e:
        print(f"[API] Note loading SCM model: {e}")


class StepRequest(BaseModel):
    T: float
    L: float
    v: float
    h: float
    U: Optional[float] = 0.0
    eps_T: Optional[float] = 0.0
    eps_L: Optional[float] = 0.0


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "KAARAN Causal World Model API",
        "endpoints": ["/health", "/simulate/step", "/predict/scm", "/artifacts/metrics"],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "scm_loaded": scm_model is not None}


@app.post("/simulate/step")
def simulate_step(req: StepRequest):
    (T_next, L_next), comp = sim.step(
        s=(req.T, req.L),
        a=(req.v, req.h),
        U=req.U or 0.0,
        eps_T=req.eps_T or 0.0,
        eps_L=req.eps_L or 0.0,
    )
    return {
        "T_next": T_next,
        "L_next": L_next,
        "delta_T": comp["delta_T"],
        "delta_L": comp["delta_L"],
    }


@app.get("/artifacts/{artifact_name}")
def get_artifact(artifact_name: str):
    path = os.path.join("web", "public", "artifacts", f"{artifact_name}.json")
    if not os.path.exists(path):
        return {"error": "Artifact not found"}
    with open(path, "r") as f:
        return json.load(f)
