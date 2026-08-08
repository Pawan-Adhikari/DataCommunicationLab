import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="FM Signal Synthesizer", layout="wide")
st.title("Interactive FM Signal Synthesizer")
st.markdown("Adjust the parameters to synthesize and visualize Frequency Modulated signals.")

# Sidebar parameters
st.sidebar.header("1. Signal Variables")
Ac = st.sidebar.slider("Carrier Amplitude (Ac)", 0.1, 3.0, 1.0, 0.1)
fc = st.sidebar.slider("Carrier Frequency (fc) Hz", 20, 500, 100, 5)
Am = st.sidebar.slider("Modulating Amplitude (Am)", 0.0, 3.0, 1.0, 0.1)
fm = st.sidebar.slider("Modulating Frequency (fm) Hz", 1, 50, 10, 1)
beta = st.sidebar.slider("Modulation Index (β)", 0.1, 10.0, 2.0, 0.1)

st.sidebar.header("2. Trace Visibility")
if "fm_trace_visibility" not in st.session_state:
    st.session_state.fm_trace_visibility = {
        "Message Signal m(t)": True,
        "Carrier": True,
        "FM Wave": True,
        "Instantaneous Frequency": True,
        "Spectrum": True,
    }

show_message = st.sidebar.checkbox("Message Signal", value=st.session_state.fm_trace_visibility["Message Signal m(t)"], key="fm_show_message")
show_carrier = st.sidebar.checkbox("Carrier", value=st.session_state.fm_trace_visibility["Carrier"], key="fm_show_carrier")
show_fm = st.sidebar.checkbox("FM Wave", value=st.session_state.fm_trace_visibility["FM Wave"], key="fm_show_fm")
show_inst_freq = st.sidebar.checkbox("Instantaneous Frequency", value=st.session_state.fm_trace_visibility["Instantaneous Frequency"], key="fm_show_inst_freq")
show_spectrum = st.sidebar.checkbox("Spectrum", value=st.session_state.fm_trace_visibility["Spectrum"], key="fm_show_spectrum")

st.session_state.fm_trace_visibility.update({
    "Message Signal m(t)": show_message,
    "Carrier": show_carrier,
    "FM Wave": show_fm,
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
fm_wave = Ac * np.sin(instantaneous_phase)
instantaneous_frequency = fc + beta * fm * np.cos(2 * np.pi * fm * t)

# Spectrum approximation using Bessel-style sidebands
freqs = np.array([fc - fm, fc, fc + fm, fc + 2 * fm, fc - 2 * fm])
amps = np.array([0.5 * beta * Ac, Ac, 0.5 * beta * Ac, 0.25 * beta * beta * Ac, 0.25 * beta * beta * Ac])

# Bessel-based spectrum approximation for FM
n_terms = int(np.ceil(beta + 6))
sideband_freqs = []
sideband_amps = []

for n in range(-n_terms, n_terms + 1):
    freq = fc + n * fm
    sideband_freqs.append(freq)
    try:
        from scipy.special import jv
    except ImportError:
        jv = None

    if jv is not None:
        amplitude = abs(jv(n, beta)) * Ac
    else:
        amplitude = Ac * np.exp(-0.5 * (n / max(beta, 1e-3)) ** 2)

    sideband_amps.append(amplitude)

# Sort and keep a clear, readable set of frequencies
order = np.argsort(sideband_freqs)
sideband_freqs = np.array(sideband_freqs)[order]
sideband_amps = np.array(sideband_amps)[order]

# Keep only significant components for display
significant_mask = sideband_amps >= max(0.05 * Ac, 0.03)
sideband_freqs = sideband_freqs[significant_mask]
sideband_amps = sideband_amps[significant_mask]

# Approximate bandwidth using significant sidebands
bandwidth = 2 * (n_terms + 1) * fm

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Time-Domain Signal", "Instantaneous Frequency and Spectrum"),
    vertical_spacing=0.18,
)

# Time-domain traces
fig.add_trace(
    go.Scatter(x=t, y=message, mode="lines", name="Message Signal m(t)", line=dict(color="purple", width=2, dash="dash"),
               visible=True if st.session_state.fm_trace_visibility["Message Signal m(t)"] else "legendonly"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=t, y=carrier, mode="lines", name="Carrier", line=dict(color="green", width=1.8, dash="dot"),
               visible=True if st.session_state.fm_trace_visibility["Carrier"] else "legendonly"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=t, y=fm_wave, mode="lines", name="FM Wave", line=dict(color="blue", width=2.2),
               visible=True if st.session_state.fm_trace_visibility["FM Wave"] else "legendonly"),
    row=1, col=1
)

# Instantaneous frequency plot in second row
fig.add_trace(
    go.Scatter(x=t, y=instantaneous_frequency, mode="lines", name="Instantaneous Frequency", line=dict(color="orange", width=2),
               visible=True if st.session_state.fm_trace_visibility["Instantaneous Frequency"] else "legendonly"),
    row=2, col=1
)

# Spectrum plot as a bar chart in the same row with a secondary y-axis is not possible with simple subplots, so we show it as a separate figure below.
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)

# Add a second figure for the spectrum
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=sideband_freqs, y=sideband_amps, name="Bessel Sidebands", marker_color="tomato"))
fig2.update_layout(
    title="FM Spectrum (Bessel Sidebands)",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Amplitude",
    height=400,
)

st.subheader("FM Bandwidth")
st.write(f"Approximate bandwidth from significant Bessel sidebands: {bandwidth:.0f} Hz")
st.write(f"Carrier frequency: {fc:.0f} Hz | Modulating frequency: {fm:.0f} Hz | Modulation index β: {beta:.2f}")
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
