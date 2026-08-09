import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.signal import hilbert
except ImportError:
    hilbert = None

try:
    from scipy.special import jv
except ImportError:
    jv = None

st.set_page_config(page_title="Analog Modulation Visualizer", layout="wide")

modulation_type = st.selectbox("Select Modulation Type", ["AM", "FM", "PM"], index=0)


def init_visibility(prefix, labels):
    key = f"{prefix}_trace_visibility"
    if key not in st.session_state:
        st.session_state[key] = {label: True for label in labels}
    return st.session_state[key]


def metric_buttons(metric_names, active_key):
    columns = st.columns(len(metric_names))
    for index, (column, metric_name) in enumerate(zip(columns, metric_names)):
        with column:
            if st.button(metric_name, key=f"{active_key}_{index}", use_container_width=True):
                if st.session_state.get(active_key) == metric_name:
                    st.session_state[active_key] = None
                else:
                    st.session_state[active_key] = metric_name
    return st.session_state.get(active_key)


def derivation_buttons(section_key, section_names):
    columns = st.columns(len(section_names))
    for index, (column, section_name) in enumerate(zip(columns, section_names)):
        with column:
            if st.button(section_name, key=f"{section_key}_{index}", use_container_width=True):
                if st.session_state.get(section_key) == section_name:
                    st.session_state[section_key] = None
                else:
                    st.session_state[section_key] = section_name
    return st.session_state.get(section_key)


def format_equation(text):
    st.markdown(text)


# -----------------------------
# AM
# -----------------------------
if modulation_type == "AM":
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

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Guide:**\n"
                        "- **Standard AM:** A1=1, A2=1, A3=1\n"
                        "- **DSB-SC:** A1=0, A2=1, A3=1\n"
                        "- **SSB (LSB):** A1=0, A2=1, A3=0\n"
                        "- **SSB (USB):** A1=0, A2=0, A3=1")

    trace_visibility = init_visibility("am", [
        "Message Signal m(t)",
        "Upper Envelope",
        "Lower Envelope",
        "Composite Waveform e(t)",
        "Carrier",
        "LSB",
        "USB",
        "Spectral Lines",
    ])

    st.sidebar.header("3. Trace Visibility")
    show_message = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="am_show_message")
    show_upper_env = st.sidebar.checkbox("Upper Envelope", value=trace_visibility["Upper Envelope"], key="am_show_upper_env")
    show_lower_env = st.sidebar.checkbox("Lower Envelope", value=trace_visibility["Lower Envelope"], key="am_show_lower_env")
    show_composite = st.sidebar.checkbox("Composite Waveform", value=trace_visibility["Composite Waveform e(t)"], key="am_show_composite")
    show_carrier = st.sidebar.checkbox("Carrier", value=trace_visibility["Carrier"], key="am_show_carrier")
    show_lsb = st.sidebar.checkbox("LSB", value=trace_visibility["LSB"], key="am_show_lsb")
    show_usb = st.sidebar.checkbox("USB", value=trace_visibility["USB"], key="am_show_usb")
    show_spectrum = st.sidebar.checkbox("Spectral Lines", value=trace_visibility["Spectral Lines"], key="am_show_spectrum")

    trace_visibility.update({
        "Message Signal m(t)": show_message,
        "Upper Envelope": show_upper_env,
        "Lower Envelope": show_lower_env,
        "Composite Waveform e(t)": show_composite,
        "Carrier": show_carrier,
        "LSB": show_lsb,
        "USB": show_usb,
        "Spectral Lines": show_spectrum,
    })

    t = np.linspace(0, 0.15, 2000)
    term1 = A1 * Ec * np.sin(2 * np.pi * fc * t)
    term2 = A2 * (Em / 2) * np.cos(2 * np.pi * (fc - fm) * t)
    term3 = -A3 * (Em / 2) * np.cos(2 * np.pi * (fc + fm) * t)
    message = Em * np.sin(2 * np.pi * fm * t)
    e_am = term1 + term2 + term3

    if hilbert is not None:
        envelope = np.abs(hilbert(e_am))
    else:
        envelope = np.abs(e_am)

    upper_envelope = envelope
    lower_envelope = -envelope

    carrier_power = ((A1 * Ec) ** 2) / 2.0 if Ec > 0 else 0.0
    sideband_power = (((A2 * Em / 2.0) ** 2) / 2.0) + (((A3 * Em / 2.0) ** 2) / 2.0)
    power_delivered = carrier_power + sideband_power
    modulation_index = Em / Ec if Ec > 0 else 0.0
    efficiency = sideband_power / power_delivered if power_delivered > 0 else 0.0
    bandwidth = 2 * fm

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Time Domain: e(t)", "Frequency Domain Spectrum"), vertical_spacing=0.15)
    fig.add_trace(go.Scatter(x=t, y=message, mode="lines", name="Message Signal m(t)", line=dict(color="purple", width=2, dash="dash"), visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=upper_envelope, mode="lines", name="Upper Envelope", line=dict(color="gray", width=1.5, dash="dash"), visible=True if trace_visibility["Upper Envelope"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=lower_envelope, mode="lines", name="Lower Envelope", line=dict(color="gray", width=1.5, dash="dash"), visible=True if trace_visibility["Lower Envelope"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=e_am, mode="lines", name="Composite Waveform e(t)", line=dict(color="blue", width=2), visible=True if trace_visibility["Composite Waveform e(t)"] else "legendonly"), row=1, col=1)
    if A1 > 0:
        fig.add_trace(go.Scatter(x=t, y=term1, mode="lines", name="Carrier", line=dict(color="green", dash="dot"), visible=True if trace_visibility["Carrier"] else "legendonly"), row=1, col=1)
    if A2 > 0:
        fig.add_trace(go.Scatter(x=t, y=term2, mode="lines", name="LSB", line=dict(color="orange", dash="dot"), visible=True if trace_visibility["LSB"] else "legendonly"), row=1, col=1)
    if A3 > 0:
        fig.add_trace(go.Scatter(x=t, y=term3, mode="lines", name="USB", line=dict(color="red", dash="dot"), visible=True if trace_visibility["USB"] else "legendonly"), row=1, col=1)

    freqs = [fm, fc - fm, fc, fc + fm]
    amps = [Em, A2 * (Em / 2), A1 * Ec, A3 * (Em / 2)]
    colors = ["purple", "orange", "green", "red"]
    fig.add_trace(go.Bar(x=freqs, y=amps, marker_color=colors, width=[2, 2, 2, 2], name="Spectral Lines", visible=True if trace_visibility["Spectral Lines"] else "legendonly"), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    ymax = max(4.5, np.max(np.abs(e_am)), np.max(envelope), Ec + Em)
    fig.update_yaxes(title_text="Voltage (V)", range=[-ymax * 1.1, ymax * 1.1], row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max(fc + fm + 50, fm + 50)], row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (V)", range=[0, max(3.5, Ec + 1, Em + 1)], row=2, col=1)
    fig.update_layout(height=700, showlegend=True, hovermode="x unified")

    st.subheader("AM Performance Metrics")
    metric_active = metric_buttons(["Power Delivered", "Modulation Index", "Efficiency", "Bandwidth"], "am_active_metric")
    if metric_active:
        with st.container(border=True):
            if metric_active == "Power Delivered":
                st.markdown("### Calculation Steps: Power Delivered")
                st.markdown(r"""
1. The carrier term is scaled by the carrier amplitude, so the carrier power is

$$
P_c = \frac{(A_1E_c)^2}{2}
$$

2. Each sideband contributes half the squared amplitude divided by 2:

$$
P_{USB} = \frac{(A_3E_m/2)^2}{2}, \qquad P_{LSB} = \frac{(A_2E_m/2)^2}{2}
$$

3. Therefore the total transmitted power is

$$
P_{AM} = P_c + P_{USB} + P_{LSB}
$$
""")
                st.write(f"Carrier power = {carrier_power:.3f} W")
                st.write(f"Upper sideband power = {((A3 * Em / 2.0) ** 2) / 2.0:.3f} W")
                st.write(f"Lower sideband power = {((A2 * Em / 2.0) ** 2) / 2.0:.3f} W")
                st.write(f"Total power delivered = {power_delivered:.3f} W")
            elif metric_active == "Modulation Index":
                st.markdown("### Calculation Steps: Modulation Index")
                st.markdown(r"""
1. The modulation index measures the ratio of message amplitude to carrier amplitude.

$$
m = \frac{E_m}{E_c}
$$

2. Substitute the current slider values to obtain the numeric result.
""")
                st.write(f"Modulation index = {modulation_index:.3f}")
                st.write(f"Using Em = {Em:.3f} and Ec = {Ec:.3f}")
            elif metric_active == "Efficiency":
                st.markdown("### Calculation Steps: Efficiency")
                st.markdown(r"""
1. Efficiency is the ratio of sideband power to total transmitted power.

$$
\eta = \frac{P_{SB}}{P_{AM}}
$$

2. For single-tone AM, sideband power is the sum of the USB and LSB powers.

$$
P_{SB} = \frac{(A_2E_m/2)^2}{2} + \frac{(A_3E_m/2)^2}{2}
$$

3. Convert the ratio into a percentage.
""")
                st.write(f"Sideband power = {sideband_power:.3f} W")
                st.write(f"Total power = {power_delivered:.3f} W")
                st.write(f"Efficiency = {efficiency * 100:.2f} %")
            elif metric_active == "Bandwidth":
                st.markdown("### Calculation Steps: Bandwidth")
                st.markdown(r"""
1. A single-tone AM spectrum has a carrier at $f_c$ and two sidebands at $f_c - f_m$ and $f_c + f_m$.

2. The occupied bandwidth is the separation between the two outer sidebands.

$$
BW = (f_c + f_m) - (f_c - f_m) = 2f_m
$$
""")
                st.write(f"Bandwidth = {bandwidth:.0f} Hz")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## About this Modulation Technique")
    st.markdown(
        """
- Amplitude Modulation keeps the carrier frequency fixed and varies only the carrier amplitude.
- The message signal appears on the envelope of the carrier, so the information is carried by changes in amplitude.
- A standard AM waveform contains a carrier and two sidebands, one above and one below the carrier frequency.
- The envelope is easy to detect, which makes AM simple to demodulate.
- If the carrier is removed, the signal becomes DSB-SC; if only one sideband remains, the signal becomes SSB.
- The occupied bandwidth is twice the highest message frequency.
- The transmitted power is split between carrier and sidebands, and the efficiency depends on the modulation index.
"""
    )
    st.subheader("Derivations")
    derivation_active = derivation_buttons("am_active_derivation", ["Eqn Derivation", "Bandwidth", "Power", "Efficiency"])
    if derivation_active:
        with st.container(border=True):
            if derivation_active == "Eqn Derivation":
                st.markdown("### Eqn Derivation")
                st.markdown(r"""
**Given:**
- $e_m = E_m\sin(\omega_m t)$
- $e_c = E_c\sin(\omega_c t)$

**Derive:**
- The general equation for an amplitude-modulated wave is created by adding the message signal to the carrier amplitude and then multiplying by the carrier wave.
- Substitute the given signals into the standard AM form:

$$
e_{AM}(t)=[E_c+E_m\sin(\omega_m t)]\sin(\omega_c t)
$$

- Distribute the terms:

$$
e_{AM}(t)=E_c\sin(\omega_c t)+E_m\sin(\omega_m t)\sin(\omega_c t)
$$

- To expand the second term, apply the standard trigonometric identity:

$$
\sin A\sin B = \frac{1}{2}[\cos(A-B)-\cos(A+B)]
$$

- Let $A=\omega_m t$ and $B=\omega_c t$:

$$
e_{AM}(t)=E_c\sin(\omega_c t)+\frac{E_m}{2}\cos(\omega_c t-\omega_m t)-\frac{E_m}{2}\cos(\omega_c t+\omega_m t)
$$

- This shows the carrier and the two sidebands at $\omega_c \pm \omega_m$.
- Note: Based on standard trigonometric identities using the provided sine waves, the signs for the lower and upper sidebands are inverted compared to the expression in the prompt. The math above reflects the exact derivation from the provided inputs.
""")
            elif derivation_active == "Bandwidth":
                st.markdown("### Derive: Bandwidth")
                st.markdown(r"""
**Derive:** Bandwidth

- From the derived equation, the signal contains three frequency components:
- Carrier frequency: $f_c$ (from $E_c\sin(\omega_c t)$)
- Lower Sideband (LSB): $f_c-f_m$ (from the $\omega_c-\omega_m$ term)
- Upper Sideband (USB): $f_c+f_m$ (from the $\omega_c+\omega_m$ term)

- The bandwidth is the difference between the highest and lowest frequencies in the signal:

$$
BW=(f_c+f_m)-(f_c-f_m)=2f_m
$$

- Therefore the occupied AM bandwidth is twice the message frequency.
""")
            elif derivation_active == "Power":
                st.markdown("### Derive: Average Power")
                st.markdown(r"""
**Derive:** Average Power in terms of $E_m$, $E_c$, and $m$

- Let $m$ be the modulation index, defined as

$$
m=\frac{E_m}{E_c}
$$

- This means $E_m=mE_c$.
- The total average power $P_{AM}$ of an AM wave across a load resistance $R$ is the sum of the carrier power $P_c$, the LSB power $P_{LSB}$, and the USB power $P_{USB}$.
- The power of a sinusoidal wave is given by

$$
P=\frac{V_{rms}^2}{R}=\frac{V_{peak}^2}{2R}
$$

- Carrier Power:

$$
P_c=\frac{E_c^2}{2R}
$$

- Sideband Voltage Peak: From the derived AM equation, the peak amplitude of each sideband is $E_m/2$, which becomes $mE_c/2$.
- Sideband Power:

$$
P_{LSB}=P_{USB}=\frac{(E_m/2)^2}{2R}=\frac{E_m^2}{8R}
$$

- Rewrite the sideband power in terms of $m$:

$$
P_{SB}=P_{LSB}+P_{USB}=\frac{E_m^2}{4R}=\frac{m^2E_c^2}{4R}=\frac{m^2}{2}P_c
$$

- Substitute back into the total power equation:

$$
P_{AM}=P_c+P_{SB}=P_c\left(1+\frac{m^2}{2}\right)=\frac{E_c^2}{2R}\left(1+\frac{m^2}{2}\right)
$$
""")
            elif derivation_active == "Efficiency":
                st.markdown("### Derive: Efficiency")
                st.markdown(r"""
**Derive:** Efficiency

- Efficiency $\eta$ is the ratio of useful power, which is the sideband power, to the total transmitted power.

$$
\eta=\frac{P_{SB}}{P_{AM}}
$$

- Using the AM power relation:

$$
P_{SB}=\frac{m^2}{2}P_c, \qquad P_{AM}=P_c\left(1+\frac{m^2}{2}\right)
$$

- Substitute these into the efficiency expression:

$$
\eta=\frac{\frac{m^2}{2}P_c}{P_c\left(1+\frac{m^2}{2}\right)}
$$

- Simplify:

$$
\eta=\frac{m^2}{m^2+2}
$$
""")

    st.markdown("### Comparative Table: AM vs FM")
    st.markdown(
        """
<table>
    <thead>
        <tr>
            <th>Feature / Aspect</th>
            <th>Amplitude Modulation (AM)</th>
            <th>Frequency Modulation (FM)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Modulation Principle</td>
            <td>Carrier amplitude varies in proportion to the modulating signal's amplitude; frequency and phase remain constant.</td>
            <td>Carrier frequency varies in proportion to the modulating signal's amplitude; amplitude and phase remain constant.</td>
        </tr>
        <tr>
            <td>Comparative Advantages</td>
            <td>
                <ul>
                    <li>Simple and inexpensive transmitter/receiver circuits.</li>
                    <li>Longer transmission range due to ground wave and sky wave propagation.</li>
                    <li>Low bandwidth requirement per channel (B = 2f_m).</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Superior noise immunity and signal-to-noise ratio (SNR).</li>
                    <li>Constant output power reduces transmitter power requirements.</li>
                    <li>High audio fidelity and dynamic range.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>Comparative Disadvantages</td>
            <td>
                <ul>
                    <li>Highly susceptible to atmospheric and electrical noise.</li>
                    <li>Inefficient power utilization (significant power consumed by the carrier).</li>
                    <li>Lower sound quality and limited dynamic range.</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Complex and costly transmitter and receiver design.</li>
                    <li>High bandwidth requirement (B = 2(Δf + f_m)).</li>
                    <li>Restricted to Line-of-Sight (LOS) range (VHF/UHF bands).</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>Bandwidth</td>
            <td>Narrow (2 × maximum modulating frequency).</td>
            <td>Wide (determined by Carson's Rule, dependent on frequency deviation).</td>
        </tr>
        <tr>
            <td>Power Distribution</td>
            <td>Power varies with modulation depth; carrier carries no information but consumes most power.</td>
            <td>Total transmitted power remains constant regardless of the modulation index.</td>
        </tr>
        <tr>
            <td>Operating Frequency</td>
            <td>Standard Medium Wave (MW: 535–1705 kHz) and Short Wave (SW: 3–30 MHz) bands.</td>
            <td>Very High Frequency (VHF: 88–108 MHz) band.</td>
        </tr>
        <tr>
            <td>Applications</td>
            <td>
                <ul>
                    <li>Commercial AM radio broadcasting.</li>
                    <li>Long-distance shortwave communications.</li>
                    <li>Air traffic control and aviation ground-to-air communications.</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>High-fidelity commercial FM music broadcasting.</li>
                    <li>TV sound transmission (analog television systems).</li>
                    <li>Marine and land mobile radio communications.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>
""",
        unsafe_allow_html=True,
    )

# -----------------------------
# FM
# -----------------------------
elif modulation_type == "FM":
    st.sidebar.header("1. Signal Variables")
    Ac = st.sidebar.slider("Carrier Amplitude (Ac)", 0.1, 3.0, 1.0, 0.1)
    fc = st.sidebar.slider("Carrier Frequency (fc) Hz", 20, 500, 100, 5)
    Am = st.sidebar.slider("Modulating Amplitude (Am)", 0.0, 3.0, 1.0, 0.1)
    fm = st.sidebar.slider("Modulating Frequency (fm) Hz", 1, 50, 10, 1)
    beta = st.sidebar.slider("Modulation Index (β)", 0.1, 10.0, 2.0, 0.1)

    trace_visibility = init_visibility("fm", [
        "Message Signal m(t)",
        "Carrier",
        "FM Wave",
        "Instantaneous Frequency",
        "Spectrum",
    ])

    st.sidebar.header("2. Trace Visibility")
    show_message = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="fm_show_message")
    show_carrier = st.sidebar.checkbox("Carrier", value=trace_visibility["Carrier"], key="fm_show_carrier")
    show_fm = st.sidebar.checkbox("FM Wave", value=trace_visibility["FM Wave"], key="fm_show_fm")
    show_inst_freq = st.sidebar.checkbox("Instantaneous Frequency", value=trace_visibility["Instantaneous Frequency"], key="fm_show_inst_freq")
    show_spectrum = st.sidebar.checkbox("Spectrum", value=trace_visibility["Spectrum"], key="fm_show_spectrum")

    trace_visibility.update({
        "Message Signal m(t)": show_message,
        "Carrier": show_carrier,
        "FM Wave": show_fm,
        "Instantaneous Frequency": show_inst_freq,
        "Spectrum": show_spectrum,
    })

    T = 0.2
    t = np.linspace(0, T, 4000)
    message = Am * np.sin(2 * np.pi * fm * t)
    carrier = Ac * np.sin(2 * np.pi * fc * t)
    instantaneous_phase = 2 * np.pi * fc * t - beta * np.cos(2 * np.pi * fm * t)
    fm_wave = Ac * np.sin(instantaneous_phase)
    instantaneous_frequency = fc + beta * fm * np.sin(2 * np.pi * fm * t)

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
    max_significant_order = int(np.max(np.abs(sideband_orders[significant_mask]))) if np.any(significant_mask) else 1
    bandwidth = 2 * max_significant_order * fm

    display_freqs = sideband_freqs[significant_mask]
    display_amps = sideband_amps[significant_mask]
    display_orders = sideband_orders[significant_mask]
    order_sort = np.argsort(display_freqs)
    display_freqs = display_freqs[order_sort]
    display_amps = display_amps[order_sort]
    display_orders = display_orders[order_sort]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Time-Domain Signal", "Instantaneous Frequency"), vertical_spacing=0.18)
    fig.add_trace(go.Scatter(x=t, y=message, mode="lines", name="Message Signal m(t)", line=dict(color="purple", width=2, dash="dash"), visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=carrier, mode="lines", name="Carrier", line=dict(color="green", width=1.8, dash="dot"), visible=True if trace_visibility["Carrier"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=fm_wave, mode="lines", name="FM Wave", line=dict(color="blue", width=2.2), visible=True if trace_visibility["FM Wave"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=instantaneous_frequency, mode="lines", name="Instantaneous Frequency", line=dict(color="orange", width=2), visible=True if trace_visibility["Instantaneous Frequency"] else "legendonly"), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, hovermode="x unified")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=display_freqs, y=display_amps, name="Bessel Sidebands", marker_color="tomato", hovertemplate="Frequency: %{x:.1f} Hz<br>Amplitude: %{y:.3f}<extra></extra>", text=[f"n={n}" for n in display_orders], textposition="outside"))
    fig2.add_vline(x=fc, line_width=2, line_dash="dash", line_color="green")
    fig2.update_layout(title="FM Spectrum (Bessel Sidebands)", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", height=420, bargap=0.25)
    fig2.update_xaxes(showgrid=True, zeroline=True)
    fig2.update_yaxes(showgrid=True, zeroline=True)

    st.subheader("FM Performance Metrics")
    fm_power = (Ac ** 2) / 2.0
    fm_metric_active = metric_buttons(["Power", "Modulation Index", "Bandwidth", "Deviation Ratio"], "fm_active_metric")
    if fm_metric_active:
        with st.container(border=True):
            if fm_metric_active == "Power":
                st.markdown("### Calculation Steps: Power")
                st.markdown(r"""
1. In FM, the carrier amplitude stays constant.

2. The average power is therefore the unmodulated carrier power.

$$
P_{FM} = \frac{E_c^2}{2}
$$
""")
                st.write(f"Average FM power = {fm_power:.3f} W")
            elif fm_metric_active == "Modulation Index":
                st.markdown("### Calculation Steps: Modulation Index")
                st.markdown(r"""
1. The FM modulation index is the ratio of peak frequency deviation to modulating frequency.

$$
\beta = \frac{\Delta f}{f_m}
$$

2. Using the current settings, the peak deviation is

$$
\Delta f = \beta f_m
$$
""")
                st.write(f"Modulation index = {beta:.3f}")
                st.write(f"Peak frequency deviation = {beta * fm:.3f} Hz")
            elif fm_metric_active == "Bandwidth":
                st.markdown("### Calculation Steps: Bandwidth")
                st.markdown(r"""
1. FM bandwidth is estimated from the significant Bessel sidebands.

2. Count the outermost order with amplitude above the threshold.

3. Use the occupied span around the carrier.

$$
BW = 2n_{max}f_m
$$
""")
                st.write(f"Largest significant sideband order = {max_significant_order}")
                st.write(f"Computed bandwidth = 2 × {max_significant_order} × {fm:.0f} = {bandwidth:.0f} Hz")
            elif fm_metric_active == "Deviation Ratio":
                st.markdown("### Calculation Steps: Deviation Ratio")
                st.markdown(r"""
1. The deviation ratio for a single-tone FM signal is the peak deviation divided by the modulating frequency.

$$
	ext{Deviation ratio} = \frac{\Delta f}{f_m}
$$

2. Since $\Delta f = \beta f_m$, the ratio is numerically the modulation index.
""")
                st.write(f"Peak frequency deviation = {beta * fm:.3f} Hz")
                st.write(f"Deviation ratio = {beta:.3f}")

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## About this Modulation Technique")
    st.markdown(
        """
- Frequency Modulation keeps the carrier amplitude constant and varies the instantaneous frequency according to the message signal.
- The message signal causes the phase to accumulate faster or slower, which changes the zero-crossing spacing of the waveform.
- FM is more resistant to amplitude noise because the information is carried by frequency deviation rather than amplitude changes.
- A single-tone FM wave produces an infinite set of sidebands, but only a finite number are significant in practice.
- The sideband amplitudes are described by Bessel functions.
- Carson's rule gives a practical occupied bandwidth estimate.
- The average transmitted power remains constant because the envelope does not change.
"""
    )
    st.subheader("Derivations")
    fm_derivation = derivation_buttons("fm_active_derivation", ["Eqn Derivation", "Bandwidth", "Power"])
    if fm_derivation:
        with st.container(border=True):
            if fm_derivation == "Eqn Derivation":
                st.markdown("### Eqn Derivation")
                st.markdown(r"""
**Given:**
- $e_m = E_m\sin(\omega_m t)$
- $e_c = E_c\sin(\omega_c t)$

**Derive:**
- In FM, the instantaneous frequency varies in direct proportion to the message signal.
- Let $k_f$ be the frequency sensitivity constant:

$$
f_i(t)=f_c+k_f e_m(t)
$$

- The total phase angle $\theta_i$ of the FM wave is the integral of the instantaneous frequency with respect to time:

$$
	theta_i(t)=\int \omega_i(t)\,dt
$$

- For a single-tone message $e_m(t)=E_m\sin(\omega_m t)$, the integral gives a cosine term.
- Let $B$ (the modulation index) be defined as

$$
B=\frac{\Delta f}{f_m}
$$

- Substituting the integrated phase into the carrier gives:

$$
	heta_i(t)=\omega_c t-B\cos(\omega_m t)+B
$$

- Substitute the instantaneous phase back into the basic carrier equation:

$$
e_{FM}(t)=E_c\sin\left(\omega_c t-B\cos(\omega_m t)+B\right)
$$

""")
            elif fm_derivation == "Bandwidth":
                st.markdown("### Derive: Bandwidth")
                st.markdown(r"""
**Derive:** Bandwidth

- Unlike AM, FM produces an infinite number of sidebands.
- However, Carson's Rule provides a derivation for the practical bandwidth that contains about 98% of the signal power.
- Carson bandwidth is written as

$$
BW\approx 2(\Delta f+f_m)
$$

- Where $\Delta f$ is the peak frequency deviation.
- We know $B=\Delta f/f_m$, which rearranges to $\Delta f=Bf_m$.
- Substitute this into Carson's rule:

$$
BW\approx 2(Bf_m+f_m)=2(B+1)f_m
$$

- Using the common notation $\beta$ for the modulation index gives the same result:

$$
BW\approx 2(\beta+1)f_m
$$
""")
            elif fm_derivation == "Power":
                st.markdown("### Derive: Average Power")
                st.markdown(r"""
**Derive:** Average Power

- In an FM wave, the amplitude $E_c$ remains strictly constant; only the frequency changes.
- Therefore, the RMS voltage of the signal is always

$$
\frac{E_c}{\sqrt{2}}
$$

- Because the envelope is constant, the total average power depends only on the unmodulated carrier amplitude and the load resistance $R$:

$$
P_{FM}=\frac{E_c^2}{2R}
$$

- The average power is constant and independent of the modulation index.
""")

    st.markdown("### Comparative Table: AM vs FM")
    st.markdown(
        """
<table>
    <thead>
        <tr>
            <th>Feature / Aspect</th>
            <th>Amplitude Modulation (AM)</th>
            <th>Frequency Modulation (FM)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Modulation Principle</td>
            <td>Carrier amplitude varies in proportion to the modulating signal's amplitude; frequency and phase remain constant.</td>
            <td>Carrier frequency varies in proportion to the modulating signal's amplitude; amplitude and phase remain constant.</td>
        </tr>
        <tr>
            <td>Comparative Advantages</td>
            <td>
                <ul>
                    <li>Simple and inexpensive transmitter/receiver circuits.</li>
                    <li>Longer transmission range due to ground wave and sky wave propagation.</li>
                    <li>Low bandwidth requirement per channel (B = 2f_m).</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Superior noise immunity and signal-to-noise ratio (SNR).</li>
                    <li>Constant output power reduces transmitter power requirements.</li>
                    <li>High audio fidelity and dynamic range.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>Comparative Disadvantages</td>
            <td>
                <ul>
                    <li>Highly susceptible to atmospheric and electrical noise.</li>
                    <li>Inefficient power utilization (significant power consumed by the carrier).</li>
                    <li>Lower sound quality and limited dynamic range.</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Complex and costly transmitter and receiver design.</li>
                    <li>High bandwidth requirement (B = 2(Δf + f_m)).</li>
                    <li>Restricted to Line-of-Sight (LOS) range (VHF/UHF bands).</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>Bandwidth</td>
            <td>Narrow (2 × maximum modulating frequency).</td>
            <td>Wide (determined by Carson's Rule, dependent on frequency deviation).</td>
        </tr>
        <tr>
            <td>Power Distribution</td>
            <td>Power varies with modulation depth; carrier carries no information but consumes most power.</td>
            <td>Total transmitted power remains constant regardless of the modulation index.</td>
        </tr>
        <tr>
            <td>Operating Frequency</td>
            <td>Standard Medium Wave (MW: 535–1705 kHz) and Short Wave (SW: 3–30 MHz) bands.</td>
            <td>Very High Frequency (VHF: 88–108 MHz) band.</td>
        </tr>
        <tr>
            <td>Applications</td>
            <td>
                <ul>
                    <li>Commercial AM radio broadcasting.</li>
                    <li>Long-distance shortwave communications.</li>
                    <li>Air traffic control and aviation ground-to-air communications.</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>High-fidelity commercial FM music broadcasting.</li>
                    <li>TV sound transmission (analog television systems).</li>
                    <li>Marine and land mobile radio communications.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>
""",
        unsafe_allow_html=True,
    )

# -----------------------------
# PM
# -----------------------------
else:
    st.sidebar.header("1. Signal Variables")
    Ac = st.sidebar.slider("Carrier Amplitude (Ac)", 0.1, 3.0, 1.0, 0.1)
    fc = st.sidebar.slider("Carrier Frequency (fc) Hz", 20, 500, 100, 5)
    Am = st.sidebar.slider("Modulating Amplitude (Am)", 0.0, 3.0, 1.0, 0.1)
    fm = st.sidebar.slider("Modulating Frequency (fm) Hz", 1, 50, 10, 1)
    beta = st.sidebar.slider("Phase Deviation Index (βp)", 0.1, 10.0, 2.0, 0.1)

    trace_visibility = init_visibility("pm", [
        "Message Signal m(t)",
        "Carrier",
        "PM Wave",
        "Instantaneous Phase",
        "Instantaneous Frequency",
        "Spectrum",
    ])

    st.sidebar.header("2. Trace Visibility")
    show_message = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="pm_show_message")
    show_carrier = st.sidebar.checkbox("Carrier", value=trace_visibility["Carrier"], key="pm_show_carrier")
    show_pm = st.sidebar.checkbox("PM Wave", value=trace_visibility["PM Wave"], key="pm_show_pm")
    show_phase = st.sidebar.checkbox("Instantaneous Phase", value=trace_visibility["Instantaneous Phase"], key="pm_show_phase")
    show_inst_freq = st.sidebar.checkbox("Instantaneous Frequency", value=trace_visibility["Instantaneous Frequency"], key="pm_show_inst_freq")
    show_spectrum = st.sidebar.checkbox("Spectrum", value=trace_visibility["Spectrum"], key="pm_show_spectrum")

    trace_visibility.update({
        "Message Signal m(t)": show_message,
        "Carrier": show_carrier,
        "PM Wave": show_pm,
        "Instantaneous Phase": show_phase,
        "Instantaneous Frequency": show_inst_freq,
        "Spectrum": show_spectrum,
    })

    T = 0.2
    t = np.linspace(0, T, 4000)
    message = Am * np.sin(2 * np.pi * fm * t)
    carrier = Ac * np.sin(2 * np.pi * fc * t)
    instantaneous_phase = 2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t)
    pm_wave = Ac * np.sin(instantaneous_phase)
    instantaneous_frequency = fc + beta * fm * np.cos(2 * np.pi * fm * t)

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
    max_significant_order = int(np.max(np.abs(sideband_orders[significant_mask]))) if np.any(significant_mask) else 1

    bessel_bandwidth = 2 * max_significant_order * fm
    carson_bandwidth = 2 * (beta + 1) * fm

    display_freqs = sideband_freqs[significant_mask]
    display_amps = sideband_amps[significant_mask]
    display_orders = sideband_orders[significant_mask]
    order_sort = np.argsort(display_freqs)
    display_freqs = display_freqs[order_sort]
    display_amps = display_amps[order_sort]
    display_orders = display_orders[order_sort]

    fig = make_subplots(rows=3, cols=1, subplot_titles=("Time-Domain Signal", "Instantaneous Phase", "Instantaneous Frequency"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=t, y=message, mode="lines", name="Message Signal m(t)", line=dict(color="purple", width=2, dash="dash"), visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=carrier, mode="lines", name="Carrier", line=dict(color="green", width=1.8, dash="dot"), visible=True if trace_visibility["Carrier"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pm_wave, mode="lines", name="PM Wave", line=dict(color="blue", width=2.2), visible=True if trace_visibility["PM Wave"] else "legendonly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=instantaneous_phase, mode="lines", name="Instantaneous Phase", line=dict(color="crimson", width=2), visible=True if trace_visibility["Instantaneous Phase"] else "legendonly"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=instantaneous_frequency, mode="lines", name="Instantaneous Frequency", line=dict(color="orange", width=2), visible=True if trace_visibility["Instantaneous Frequency"] else "legendonly"), row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_layout(height=860, showlegend=True, hovermode="x unified")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=display_freqs, y=display_amps, name="Bessel Sidebands", marker_color="tomato", hovertemplate="Frequency: %{x:.1f} Hz<br>Amplitude: %{y:.3f}<extra></extra>", text=[f"n={n}" for n in display_orders], textposition="outside"))
    fig2.add_vline(x=fc, line_width=2, line_dash="dash", line_color="green")
    fig2.update_layout(title="PM Spectrum (Bessel Sidebands)", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", height=420, bargap=0.25)
    fig2.update_xaxes(showgrid=True, zeroline=True)
    fig2.update_yaxes(showgrid=True, zeroline=True)

    st.subheader("PM Performance Metrics")
    pm_power = (Ac ** 2) / 2.0
    pm_metric_active = metric_buttons(["Power", "Phase Deviation Index", "Bandwidth", "Deviation Ratio"], "pm_active_metric")
    if pm_metric_active:
        with st.container(border=True):
            if pm_metric_active == "Power":
                st.markdown("### Calculation Steps: Power")
                st.markdown(r"""
1. In PM, the carrier amplitude stays constant.

2. Therefore the average power is the same as the unmodulated carrier.

$$
P_{PM} = \frac{E_c^2}{2}
$$
""")
                st.write(f"Average PM power = {pm_power:.3f} W")
            elif pm_metric_active == "Phase Deviation Index":
                st.markdown("### Calculation Steps: Phase Deviation Index")
                st.markdown(r"""
1. The phase deviation index is the maximum phase shift imposed by the message.

$$
\beta_p = k_p A_m
$$

2. In this visualizer, the slider value is used directly as the phase deviation index.
""")
                st.write(f"Phase deviation index = {beta:.3f}")
            elif pm_metric_active == "Bandwidth":
                st.markdown("### Calculation Steps: Bandwidth")
                st.markdown(r"""
1. PM also produces Bessel-weighted sidebands around the carrier.

2. The occupied bandwidth is estimated from the highest significant sideband order.

3. The total span is twice the maximum offset from the carrier.

$$
BW = 2n_{max}f_m
$$
""")
                st.write(f"Largest significant sideband order = {max_significant_order}")
                st.write(f"Computed bandwidth = 2 × {max_significant_order} × {fm:.0f} = {bessel_bandwidth:.0f} Hz")
            elif pm_metric_active == "Deviation Ratio":
                st.markdown("### Calculation Steps: Deviation Ratio")
                st.markdown(r"""
1. The PM deviation scale can be compared using the product of phase index and modulating frequency.

$$
\beta_p f_m
$$

2. This gives a practical indicator of the strength of phase variation.
""")
                st.write(f"Approximate frequency deviation scale = {beta * fm:.3f} Hz")

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## About this Modulation Technique")
    st.markdown(
        """
- Phase Modulation keeps the carrier amplitude constant and varies only the instantaneous phase of the carrier.
- The message signal is mapped directly to a phase shift, which changes the timing of the carrier oscillation.
- For a single-tone message, PM produces Bessel-weighted sidebands similar to FM.
- The spectrum becomes wider as the phase deviation index increases.
- Because the envelope remains constant, the average power does not change with modulation.
- PM is often treated as the phase-domain counterpart of FM for a single-tone input.
- The bandwidth is commonly estimated using the same Carson-style rule used for FM.
"""
    )
    st.subheader("Derivations")
    pm_derivation = derivation_buttons("pm_active_derivation", ["Eqn Derivation", "Bandwidth", "Power"])
    if pm_derivation:
        with st.container(border=True):
            if pm_derivation == "Eqn Derivation":
                st.markdown("### Eqn Derivation")
                st.markdown(r"""
**Given:**
- $e_m = E_m\sin(\omega_m t)$
- $e_c = E_c\sin(\omega_c t)$

**Derive:**
- In PM, the instantaneous phase varies in direct proportion to the message signal.
- Let $k_p$ be the phase sensitivity constant:

$$
	heta_i(t)=\omega_c t+k_p e_m(t)
$$

- Let the PM modulation index be

$$
\beta_p=k_pE_m
$$

- Substituting this gives the total phase:

$$
	heta_i(t)=\omega_c t+\beta_p\sin(\omega_m t)
$$

- Substitute the total phase into the basic carrier equation:

$$
e_{PM}(t)=E_c\sin\left(\omega_c t+\beta_p\sin(\omega_m t)\right)
$$

- This is equivalent in form to FM for a single-tone message, so its spectrum is also described by Bessel-weighted sidebands.
""")
            elif pm_derivation == "Bandwidth":
                st.markdown("### Derive: Bandwidth")
                st.markdown(r"""
**Derive:** Bandwidth

- To find the bandwidth using Carson's Rule, first determine the peak frequency deviation $\Delta f$.
- Instantaneous frequency is the derivative of instantaneous phase:

$$
\omega_i(t)=\frac{d\theta_i(t)}{dt}=\omega_c+\beta_p\omega_m\cos(\omega_m t)
$$

- The peak angular frequency deviation is $\beta_p\omega_m$, which converts to peak frequency deviation

$$
\Delta f=\beta_p f_m
$$

- Applying Carson's Rule:

$$
BW\approx 2(\Delta f+f_m)
$$

- Substitute $\Delta f=\beta_p f_m$:

$$
BW\approx 2(\beta_p+1)f_m
$$
""")
            elif pm_derivation == "Power":
                st.markdown("### Derive: Average Power")
                st.markdown(r"""
**Derive:** Average Power

- Like Frequency Modulation, a Phase Modulated wave has a constant amplitude envelope $E_c$.
- Because the amplitude never changes, the power does not fluctuate with the modulating signal.
- Therefore the average power is the same as the unmodulated carrier:

$$
P_{PM}=\frac{E_c^2}{2R}
$$

- The average power is constant and independent of the phase deviation index.
""")

    st.markdown("### Phase Modulation (PM)")
    st.markdown(
        """
**Advantages**

⚬ High Noise Immunity: Highly resistant to amplitude-based atmospheric and electrical noise.

⚬ Constant Power Output: Total transmitted power remains constant, allowing efficient amplifier operation.

⚬ Superior SNR: Delivers a significantly better Signal-to-Noise Ratio than AM.

⚬ Foundation for Digital Tech: Easily adapted into robust digital communication schemes (e.g., PSK, QAM).

**Disadvantages**

⚬ High Bandwidth Requirement: Requires significantly more channel bandwidth than AM, which expands at higher signal frequencies.

⚬ Circuit Complexity: Hardware design for phase modulators and demodulators (PLLs) is complex and expensive.

⚬ Phase Ambiguity: Phase shifts in the transmission channel can cause synchronization loss and demodulation errors at the receiver.

**Applications**

⚬ Digital Wireless Communications: Core technology behind Wi-Fi, Bluetooth, and cellular networks (4G/5G) via BPSK, QPSK, and QAM.

⚬ Satellite & Deep-Space Telemetry: Used in satellite data links and space probes for reliable, low-noise data transfer.

⚬ Digital Sound Synthesis: Utilized in electronic music synthesizers (e.g., Yamaha DX series phase distortion synthesis).

⚬ Radar & Military Systems: Applied in phase-array radars and pulse compression systems to measure precise target distances and velocity.
"""
    )
