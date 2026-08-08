import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.signal import hilbert
except ImportError:
    hilbert = None

# Page config
st.set_page_config(page_title="AM Signal Synthesizer", layout="wide")
st.title("Interactive AM Signal Synthesizer")
st.markdown("Adjust the variables and toggle the equation terms to synthesize Standard AM, DSB-SC, or SSB signals.")

# Sidebar for Parameters
st.sidebar.header("1. Signal Variables")
Ec = st.sidebar.slider("Carrier Amplitude (Ec)", 0.1, 3.0, 1.0, 0.1)
fc = st.sidebar.slider("Carrier Frequency (fc) Hz", 20, 300, 100, 5)
Em = st.sidebar.slider("Modulating Amplitude (Em)", 0.0, 3.0, 0.8, 0.1)
fm = st.sidebar.slider("Modulating Freq (fm) Hz", 1, 30, 10, 1)

st.sidebar.header("2. Term Weights (On/Off)")
st.sidebar.markdown("Set to 1 to include, 0 to suppress.")
A1 = st.sidebar.slider("A1: Term 1 (Carrier)", 0.0, 1.0, 1.0, step=1.0)
A2 = st.sidebar.slider("A2: Term 2 (Lower Sideband)", 0.0, 1.0, 1.0, step=1.0)
A3 = st.sidebar.slider("A3: Term 3 (Upper Sideband)", 0.0, 1.0, 1.0, step=1.0)

# Quick Presets Information
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Guide:**\n"
                    "- **Standard AM:** A1=1, A2=1, A3=1\n"
                    "- **DSB-SC:** A1=0, A2=1, A3=1\n"
                    "- **SSB (LSB):** A1=0, A2=1, A3=0\n"
                    "- **SSB (USB):** A1=0, A2=0, A3=1")

# Persistent trace visibility state
if "trace_visibility" not in st.session_state:
    st.session_state.trace_visibility = {
        "Message Signal m(t)": True,
        "Upper Envelope": True,
        "Lower Envelope": True,
        "Composite Waveform e(t)": True,
        "Carrier": True,
        "LSB": True,
        "USB": True,
        "Spectral Lines": True,
    }

st.sidebar.header("3. Trace Visibility")
show_message = st.sidebar.checkbox("Message Signal", value=st.session_state.trace_visibility["Message Signal m(t)"], key="show_message")
show_upper_env = st.sidebar.checkbox("Upper Envelope", value=st.session_state.trace_visibility["Upper Envelope"], key="show_upper_env")
show_lower_env = st.sidebar.checkbox("Lower Envelope", value=st.session_state.trace_visibility["Lower Envelope"], key="show_lower_env")
show_composite = st.sidebar.checkbox("Composite Waveform", value=st.session_state.trace_visibility["Composite Waveform e(t)"], key="show_composite")
show_carrier = st.sidebar.checkbox("Carrier", value=st.session_state.trace_visibility["Carrier"], key="show_carrier")
show_lsb = st.sidebar.checkbox("LSB", value=st.session_state.trace_visibility["LSB"], key="show_lsb")
show_usb = st.sidebar.checkbox("USB", value=st.session_state.trace_visibility["USB"], key="show_usb")
show_spectrum = st.sidebar.checkbox("Spectral Lines", value=st.session_state.trace_visibility["Spectral Lines"], key="show_spectrum")

st.session_state.trace_visibility.update({
    "Message Signal m(t)": show_message,
    "Upper Envelope": show_upper_env,
    "Lower Envelope": show_lower_env,
    "Composite Waveform e(t)": show_composite,
    "Carrier": show_carrier,
    "LSB": show_lsb,
    "USB": show_usb,
    "Spectral Lines": show_spectrum,
})

# Time Array
t = np.linspace(0, 0.15, 2000)

# Equation Components
term1 = A1 * Ec * np.sin(2 * np.pi * fc * t)                       # Carrier
term2 = A2 * (Em / 2) * np.cos(2 * np.pi * (fc - fm) * t)          # LSB
term3 = -A3 * (Em / 2) * np.cos(2 * np.pi * (fc + fm) * t)         # USB
message = Em * np.sin(2 * np.pi * fm * t)                           # Message signal

# Composite Signal
e_AM = term1 + term2 + term3

# Envelope estimation for the AM waveform
if hilbert is not None:
    analytic_signal = hilbert(e_AM)
    envelope = np.abs(analytic_signal)
else:
    envelope = np.abs(e_AM)

upper_envelope = envelope
lower_envelope = -envelope

# AM metrics for display
carrier_power = ((A1 * Ec) ** 2) / 2.0 if Ec > 0 else 0.0
sideband_power = (((A2 * Em / 2.0) ** 2) / 2.0) + (((A3 * Em / 2.0) ** 2) / 2.0)
power_delivered = carrier_power + sideband_power
modulation_index = Em / Ec if Ec > 0 else 0.0
efficiency = sideband_power / power_delivered if power_delivered > 0 else 0.0
bandwidth = 2 * fm

# Create Plotly Figure with Subplots (Time and Frequency)
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=("Time Domain: e(t)", "Frequency Domain Spectrum"),
    vertical_spacing=0.15
)

# Time Domain Traces
fig.add_trace(go.Scatter(x=t, y=message, mode='lines', name='Message Signal m(t)', line=dict(color='purple', width=2, dash='dash'), visible=True if st.session_state.trace_visibility['Message Signal m(t)'] else 'legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=upper_envelope, mode='lines', name='Upper Envelope', line=dict(color='gray', width=1.5, dash='dash'), visible=True if st.session_state.trace_visibility['Upper Envelope'] else 'legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=lower_envelope, mode='lines', name='Lower Envelope', line=dict(color='gray', width=1.5, dash='dash'), visible=True if st.session_state.trace_visibility['Lower Envelope'] else 'legendonly'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=e_AM, mode='lines', name='Composite Waveform e(t)', line=dict(color='blue', width=2), visible=True if st.session_state.trace_visibility['Composite Waveform e(t)'] else 'legendonly'), row=1, col=1)
if A1 > 0: fig.add_trace(go.Scatter(x=t, y=term1, mode='lines', name='Carrier', line=dict(color='green', dash='dot'), visible=True if st.session_state.trace_visibility['Carrier'] else 'legendonly'), row=1, col=1)
if A2 > 0: fig.add_trace(go.Scatter(x=t, y=term2, mode='lines', name='LSB', line=dict(color='orange', dash='dot'), visible=True if st.session_state.trace_visibility['LSB'] else 'legendonly'), row=1, col=1)
if A3 > 0: fig.add_trace(go.Scatter(x=t, y=term3, mode='lines', name='USB', line=dict(color='red', dash='dot'), visible=True if st.session_state.trace_visibility['USB'] else 'legendonly'), row=1, col=1)

# Frequency Domain Traces (Spectrum)
freqs = [fm, fc - fm, fc, fc + fm]
amps = [Em, A2 * (Em / 2), A1 * Ec, A3 * (Em / 2)]
colors = ['purple', 'orange', 'green', 'red']

fig.add_trace(
    go.Bar(x=freqs, y=amps, marker_color=colors, width=[2, 2, 2, 2], name='Spectral Lines', visible=True if st.session_state.trace_visibility['Spectral Lines'] else 'legendonly'),
    row=2, col=1
)

# Update Layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
ymax = max(4.5, np.max(np.abs(e_AM)), np.max(envelope), Ec + Em)
fig.update_yaxes(title_text="Voltage (V)", range=[-ymax * 1.1, ymax * 1.1], row=1, col=1)

fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max(fc + fm + 50, fm + 50)], row=2, col=1)
fig.update_yaxes(title_text="Amplitude (V)", range=[0, max(3.5, Ec + 1, Em + 1)], row=2, col=1)

fig.update_layout(height=700, showlegend=True, hovermode="x unified")

# Render metrics above the plot
st.subheader("AM Performance Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    if st.button("Power Delivered", key="metric_power", use_container_width=True):
        st.session_state["active_metric"] = "Power Delivered"
with metric_col2:
    if st.button("Modulation Index", key="metric_mod", use_container_width=True):
        st.session_state["active_metric"] = "Modulation Index"
with metric_col3:
    if st.button("Efficiency", key="metric_eff", use_container_width=True):
        st.session_state["active_metric"] = "Efficiency"
with metric_col4:
    if st.button("Bandwidth", key="metric_bw", use_container_width=True):
        st.session_state["active_metric"] = "Bandwidth"

if "active_metric" not in st.session_state:
    st.session_state["active_metric"] = "Power Delivered"

st.markdown("---")

if st.session_state["active_metric"] == "Power Delivered":
    with st.expander("Calculation Steps: Power Delivered", expanded=True):
        st.write(f"Carrier power = $\\frac{{(A_1 E_c)^2}}{2}$ = {carrier_power:.3f} W")
        st.write(f"Sideband power = $\\frac{{(A_2 E_m / 2)^2}}{2}$ + $\\frac{{(A_3 E_m / 2)^2}}{2}$ = {sideband_power:.3f} W")
        st.write(f"Total power delivered = carrier power + sideband power = {power_delivered:.3f} W")
elif st.session_state["active_metric"] == "Modulation Index":
    with st.expander("Calculation Steps: Modulation Index", expanded=True):
        st.write(f"Modulation index $m = \\frac{{E_m}}{{E_c}}$ = {modulation_index:.3f}")
elif st.session_state["active_metric"] == "Efficiency":
    with st.expander("Calculation Steps: Efficiency", expanded=True):
        st.write(f"Efficiency $\\eta = \\frac{{sideband power}}{{total power}}$ = {efficiency * 100:.2f} %")
elif st.session_state["active_metric"] == "Bandwidth":
    with st.expander("Calculation Steps: Bandwidth", expanded=True):
        st.write(f"Bandwidth $BW = 2f_m$ = {bandwidth:.0f} Hz")

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)
