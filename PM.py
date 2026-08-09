import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.special import jv
except ImportError:
    jv = None

st.set_page_config(page_title="PM Signal Synthesizer", layout="wide")
st.title("Interactive PM Signal Synthesizer")
st.markdown("Adjust the parameters to synthesize and visualize Phase Modulated signals.")

# Sidebar parameters
st.sidebar.header("1. Signal Variables")
Ac = st.sidebar.slider("Carrier Amplitude (Ac)", 0.1, 3.0, 1.0, 0.1)
fc = st.sidebar.slider("Carrier Frequency (fc) Hz", 20, 500, 100, 5)
Am = st.sidebar.slider("Modulating Amplitude (Am)", 0.0, 3.0, 1.0, 0.1)
fm = st.sidebar.slider("Modulating Frequency (fm) Hz", 1, 50, 10, 1)
beta = st.sidebar.slider("Phase Deviation Index (βp)", 0.1, 10.0, 2.0, 0.1)

st.sidebar.header("2. Trace Visibility")
if "pm_trace_visibility" not in st.session_state:
    st.session_state.pm_trace_visibility = {
        "Message Signal m(t)": True,
        "Carrier": True,
        "PM Wave": True,
        "Instantaneous Phase": True,
        "Instantaneous Frequency": True,
        "Spectrum": True,
    }

show_message = st.sidebar.checkbox("Message Signal", value=st.session_state.pm_trace_visibility["Message Signal m(t)"], key="pm_show_message")
show_carrier = st.sidebar.checkbox("Carrier", value=st.session_state.pm_trace_visibility["Carrier"], key="pm_show_carrier")
show_pm = st.sidebar.checkbox("PM Wave", value=st.session_state.pm_trace_visibility["PM Wave"], key="pm_show_pm")
show_phase = st.sidebar.checkbox("Instantaneous Phase", value=st.session_state.pm_trace_visibility["Instantaneous Phase"], key="pm_show_phase")
show_inst_freq = st.sidebar.checkbox("Instantaneous Frequency", value=st.session_state.pm_trace_visibility["Instantaneous Frequency"], key="pm_show_inst_freq")
show_spectrum = st.sidebar.checkbox("Spectrum", value=st.session_state.pm_trace_visibility["Spectrum"], key="pm_show_spectrum")

st.session_state.pm_trace_visibility.update({
    "Message Signal m(t)": show_message,
    "Carrier": show_carrier,
    "PM Wave": show_pm,
    "Instantaneous Phase": show_phase,
    "Instantaneous Frequency": show_inst_freq,
    "Spectrum": show_spectrum,
})

# Time axis
T = 0.2
t = np.linspace(0, T, 4000)

# Signals
message = Am * np.sin(2 * np.pi * fm * t)
carrier = Ac * np.sin(2 * np.pi * fc * t)
instantaneous_phase = 2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t)
pm_wave = Ac * np.sin(instantaneous_phase)
instantaneous_frequency = fc + beta * fm * np.cos(2 * np.pi * fm * t)

# Spectral expansion using Bessel coefficients
n_terms = int(np.ceil(beta + 8))
sideband_orders = []
sideband_freqs = []
sideband_amps = []

for n in range(-n_terms, n_terms + 1):
    freq = fc + n * fm
    if jv is not None:
        amplitude = abs(jv(n, beta)) * Ac
    else:
        amplitude = Ac * np.exp(-0.5 * (n / max(beta, 1e-3)) ** 2)

    sideband_orders.append(n)
    sideband_freqs.append(freq)
    sideband_amps.append(amplitude)

sideband_orders = np.array(sideband_orders)
sideband_freqs = np.array(sideband_freqs)
sideband_amps = np.array(sideband_amps)

significant_threshold = max(0.01 * Ac, 0.02)
significant_mask = sideband_amps >= significant_threshold

if np.any(significant_mask):
    max_significant_order = int(np.max(np.abs(sideband_orders[significant_mask])))
else:
    max_significant_order = 1

bessel_bandwidth = 2 * max_significant_order * fm
carson_bandwidth = 2 * (beta + 1) * fm

display_freqs = sideband_freqs[significant_mask]
display_amps = sideband_amps[significant_mask]
display_orders = sideband_orders[significant_mask]

order_sort = np.argsort(display_freqs)
display_freqs = display_freqs[order_sort]
display_amps = display_amps[order_sort]
display_orders = display_orders[order_sort]

# Time-domain figure with separate phase and frequency panes
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Time-Domain Signal", "Instantaneous Phase", "Instantaneous Frequency"),
    vertical_spacing=0.12,
)

fig.add_trace(
    go.Scatter(
        x=t,
        y=message,
        mode="lines",
        name="Message Signal m(t)",
        line=dict(color="purple", width=2, dash="dash"),
        visible=True if st.session_state.pm_trace_visibility["Message Signal m(t)"] else "legendonly",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=t,
        y=carrier,
        mode="lines",
        name="Carrier",
        line=dict(color="green", width=1.8, dash="dot"),
        visible=True if st.session_state.pm_trace_visibility["Carrier"] else "legendonly",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=t,
        y=pm_wave,
        mode="lines",
        name="PM Wave",
        line=dict(color="blue", width=2.2),
        visible=True if st.session_state.pm_trace_visibility["PM Wave"] else "legendonly",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=t,
        y=instantaneous_phase,
        mode="lines",
        name="Instantaneous Phase",
        line=dict(color="crimson", width=2),
        visible=True if st.session_state.pm_trace_visibility["Instantaneous Phase"] else "legendonly",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=t,
        y=instantaneous_frequency,
        mode="lines",
        name="Instantaneous Frequency",
        line=dict(color="orange", width=2),
        visible=True if st.session_state.pm_trace_visibility["Instantaneous Frequency"] else "legendonly",
    ),
    row=3,
    col=1,
)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
fig.update_layout(height=720, showlegend=True, hovermode="x unified")

# Spectrum figure
fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=display_freqs,
        y=display_amps,
        name="Bessel Sidebands",
        marker_color="tomato",
        hovertemplate="Frequency: %{x:.1f} Hz<br>Amplitude: %{y:.3f}<extra></extra>",
        text=[f"n={n}" for n in display_orders],
        textposition="outside",
    )
)
fig2.add_vline(x=fc, line_width=2, line_dash="dash", line_color="green")
fig2.update_layout(
    title="PM Spectrum (Bessel Sidebands)",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Amplitude",
    height=420,
    bargap=0.25,
)
fig2.update_xaxes(showgrid=True, zeroline=True)
fig2.update_yaxes(showgrid=True, zeroline=True)

st.subheader("PM Bandwidth")
st.write(f"Bandwidth from significant Bessel sidebands (|J_n(βp)| >= {significant_threshold:.3f}): {bessel_bandwidth:.0f} Hz")
st.write(f"Carson's rule estimate: {carson_bandwidth:.0f} Hz")
st.write(f"Carrier frequency: {fc:.0f} Hz | Modulating frequency: {fm:.0f} Hz | Phase deviation index βp: {beta:.2f}")
st.write("The message signal stays sine, and the instantaneous phase follows the same sine reference.")

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Fourier Transform Representation and Derivation")
st.markdown(
    r"""
For a single-tone PM signal with message $m(t)=A_m\sin(\omega_m t)$,

$$
s_{PM}(t)=A_c\cos\left(\omega_c t+\beta_p\sin(\omega_m t)\right)
$$

where $\beta_p=k_pA_m$ is the phase deviation index.

Using the Jacobi-Anger expansion,

$$
e^{j\beta_p\sin\theta}=\sum_{n=-\infty}^{\infty}J_n(\beta_p)e^{jn\theta}
$$

the PM waveform becomes

$$
s_{PM}(t)=A_c\sum_{n=-\infty}^{\infty}J_n(\beta_p)\cos\left((\omega_c+n\omega_m)t\right)
$$

Hence the spectrum has discrete components at $\omega_c+n\omega_m$ with weights $J_n(\beta_p)$:

$$
S_{PM}(\omega)=\pi A_c\sum_{n=-\infty}^{\infty}J_n(\beta_p)\left[\delta(\omega-(\omega_c+n\omega_m))+\delta(\omega+(\omega_c+n\omega_m))\right]
$$
"""
)
