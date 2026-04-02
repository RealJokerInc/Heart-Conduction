# Ionic Surrogate v3 — Stage 1

::: mermaid
graph TD;
    CS[carried_state t 20T] --> ATTN[VoltageAttention<br/>20 dims to Vm dt];
    VM[Vm dt] --> ATTN;
    ATTN --> SPLIT{SPLIT};
    SPLIT -->|ionic 16| IONIC[ionic_mid 16T];
    SPLIT -->|conc 4| CONC[conc t+1 4T];
    IONIC --> RMS[rms_norm];
    RMS --> MLP[ionic_mixing_mlp<br/>16 to 16 to 16];
    MLP --> INTERP1[interpolate<br/>ionic_mixing_logit 16p];
    IONIC -.->|z_mid skip| INTERP1;
    INTERP1 --> IONICOUT[ionic_state t+1 16T];
    IONICOUT --> RECOMBINE[RECOMBINE];
    CONC --> RECOMBINE;
    RECOMBINE --> CSOUT[carried_state t+1 20T];
    CSOUT --> GCLIN[gate_conductance_linear<br/>20 to 8];
    CSOUT --> GCMLP[gate_conductance_mlp<br/>20 to 12 to 12 to 8];
    GCLIN --> INTERP2[interpolate<br/>gate_conductance_logit 8p];
    GCMLP --> INTERP2;
    INTERP2 --> CONDOUT[cond_lat t+1 8T];
    CONC --> NERNST[NernstComputer<br/>fixed physics 0p];
    NERNST --> NSTOUT[nernst_st t+1 8T];
    IONICOUT -.->|scaffold| SDEC1[ionic_state_decoder 16 to 15];
    CONDOUT -.->|scaffold| SDEC2[gate_conductance_decoder 8 to 5];
    CONC -.->|scaffold| CMSE[direct MSE];
:::

# Ionic Surrogate v3 — Stage 2

::: mermaid
graph TD;
    CONDR[cond_lat t 8T] --> EQ[e_q 8x4];
    NSTR[nernst_st t 8T] --> NORM[normalize<br/>env 9T];
    VMR[Vm] --> NORM;
    NORM --> EK[e_k 9x4];
    NORM --> EV[e_v 9x1];
    EQ --> SCORES[ConductanceAttention<br/>QKT div sqrt4<br/>NO softmax];
    EK --> SCORES;
    SCORES --> ATT[scores x V<br/>attended 8];
    EV --> ATT;
    ATT --> OMLP[output_mlp<br/>8 to 4 GELU to 1];
    OMLP --> IION[I_ion t scalar];
:::
