%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#4A90D9', 'primaryTextColor': '#fff', 'primaryBorderColor': '#3570A8', 'lineColor': '#555', 'secondaryColor': '#E8913A', 'tertiaryColor': '#2EAD6B'}}}%%

graph TD
    subgraph Stage1["<b>Stage 1: State Evolution</b><br/><i>off critical path · 1,416 params</i>"]
        direction TB

        CS["carried_state(t)<br/>20ᵀ = ionic(16) | conc(4)"]
        VM["Vm, dt"]

        ATTN["<b>VoltageAttention</b><br/>20 dims attend to [Vm,dt]<br/>σ(Q·Kᵀ/√4) · contractive<br/><i>W_q(20,4) W_k(2,4) W_v(2,4) W_out(4,20)</i>"]

        VM --> ATTN
        CS --> ATTN
        CS -.->|"skip<br/>(prev + residual)"| HAD
        CS -.->|"skip"| ADD

        ATTN --> SPLIT

        SPLIT{"SPLIT after attention"}

        SPLIT -->|"[:16]"| IONIC["ionic_mid<br/>16ᵀ"]
        SPLIT -->|"[16:]"| CONC["conc(t+1) 4ᵀ<br/><i>attention only — DONE</i>"]

        IONIC --> RMS["rms_norm<br/><i>0 params</i>"]
        RMS --> MLP["<b>ionic_mixing_mlp</b><br/>Lin(16,16) → GELU → Lin(16,16)"]

        HAD(("⊗"))
        ADD(("⊕"))

        MLP --> HAD
        HAD -->|"gate·(target−prev)"| ADD
        ADD -->|"z_mid"| INTERP1

        INTERP1["<b>interpolate</b><br/>(1−α)·z_mid + α·correction<br/><i>ionic_mixing_logit: 16 params</i>"]

        INTERP1 --> IONICOUT["ionic_state(t+1)<br/>16ᵀ"]
        IONICOUT --> RECOMBINE["RECOMBINE<br/>cat(ionic, conc)"]
        CONC --> RECOMBINE

        RECOMBINE --> CSOUT["carried_state(t+1)<br/>20ᵀ"]

        CSOUT --> GCLIN["<b>gate_conductance_linear</b><br/>Lin(20,8)"]
        CSOUT --> GCMLP["<b>gate_conductance_mlp</b><br/>Lin(20,12)→GELU→Lin(12,12)→GELU→Lin(12,8)"]

        GCLIN --> INTERP2
        GCMLP --> INTERP2
        INTERP2["<b>interpolate</b><br/>(1−β)·linear + β·nonlinear<br/><i>gate_conductance_logit: 8 params</i>"]

        INTERP2 --> CONDOUT["cond_lat(t+1)<br/>8ᵀ"]

        CONC --> NERNST["<b>NernstComputer</b><br/>fixed physics, 0 params<br/>E_Na, E_K, E_Ca, E_Ks"]
        NERNST --> NSTOUT["nernst_st(t+1)<br/>8ᵀ"]
    end

    subgraph Stage2["<b>Stage 2: Current Readout</b><br/><i>ON critical path · 118 params</i>"]
        direction TB

        CONDR["cond_lat(t) 8ᵀ"]
        NSTR["nernst_st(t) 8ᵀ"]
        VMR["Vm"]

        NSTR --> NORM["<b>normalize</b><br/>nernst_st + Vm → env 9ᵀ<br/><i>fixed shift/scale</i>"]
        VMR --> NORM

        CONDR --> EQ["<b>e_q</b> (8,4)"]
        NORM --> EK["<b>e_k</b> (9,4)"]
        NORM --> EV["<b>e_v</b> (9,1)"]

        EQ --> SCORES
        EK --> SCORES
        SCORES["<b>ConductanceAttention</b><br/>QKᵀ/√4 → scores (8,9)<br/><i>NO softmax</i>"]

        SCORES --> ATTENDED
        EV --> ATTENDED
        ATTENDED["scores × V<br/>attended (8,)"]

        ATTENDED --> OMLP["<b>output_mlp</b><br/>Lin(8,4) → GELU → Lin(4,1)"]
        OMLP --> IION["<b>I_ion(t)</b><br/>scalar"]
    end

    subgraph Scaffolds["<b>Scaffolds</b> <i>(training only)</i>"]
        direction LR
        SDEC1["ionic_state_decoder<br/>Lin(16,15)"]
        SDEC2["gate_conductance_decoder<br/>Lin(8,5)"]
        CMSE["direct MSE<br/>vs true conc"]
    end

    IONICOUT -.-> SDEC1
    CONDOUT -.-> SDEC2
    CONC -.-> CMSE

    style Stage1 fill:#f0f4ff,stroke:#4A90D9,stroke-width:2px
    style Stage2 fill:#fff0f0,stroke:#D94A5E,stroke-width:2px
    style Scaffolds fill:#f5f5f5,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5

    style CS fill:#E8D5F5,stroke:#6C5CE7
    style VM fill:#E8D5F5,stroke:#6C5CE7
    style CONDR fill:#E8D5F5,stroke:#6C5CE7
    style NSTR fill:#E8D5F5,stroke:#6C5CE7
    style VMR fill:#E8D5F5,stroke:#6C5CE7

    style ATTN fill:#B8D4F0,stroke:#4A90D9
    style IONIC fill:#B8D4F0,stroke:#4A90D9
    style MLP fill:#FDDCB5,stroke:#E8913A
    style INTERP1 fill:#FDDCB5,stroke:#E8913A
    style IONICOUT fill:#FDDCB5,stroke:#E8913A

    style GCLIN fill:#C8E6C9,stroke:#2EAD6B
    style GCMLP fill:#C8E6C9,stroke:#2EAD6B
    style INTERP2 fill:#C8E6C9,stroke:#2EAD6B
    style CONDOUT fill:#C8E6C9,stroke:#2EAD6B

    style CONC fill:#E1BEE7,stroke:#8E6CC0
    style NERNST fill:#E1BEE7,stroke:#8E6CC0
    style NSTOUT fill:#E1BEE7,stroke:#8E6CC0

    style SCORES fill:#FFCDD2,stroke:#D94A5E
    style ATTENDED fill:#FFCDD2,stroke:#D94A5E
    style OMLP fill:#FFCDD2,stroke:#D94A5E
    style IION fill:#FFCDD2,stroke:#D94A5E,stroke-width:3px
    style NORM fill:#FFCDD2,stroke:#D94A5E

    style SDEC1 fill:#f5f5f5,stroke:#999,stroke-dasharray: 3 3
    style SDEC2 fill:#f5f5f5,stroke:#999,stroke-dasharray: 3 3
    style CMSE fill:#f5f5f5,stroke:#999,stroke-dasharray: 3 3
