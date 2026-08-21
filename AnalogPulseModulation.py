import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    butter = None
    filtfilt = None

st.set_page_config(page_title="Analog Pulse Modulation Visualizer", layout="wide")


# -----------------------------
# Helper Functions
# -----------------------------
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


def apply_lpf(signal, fs, cutoff=15.0, order=4):
    """Low-pass filter signal to reconstruct message."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.8
    if butter is not None and filtfilt is not None and cutoff > 0:
        b, a = butter(order, cutoff / nyq, btype='low')
        filtered = filtfilt(b, a, signal)
        return filtered
    else:
        # Fallback ideal LPF via FFT
        N = len(signal)
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, 1.0 / fs)
        fft_vals[freqs > cutoff] = 0.0
        return np.fft.irfft(fft_vals, n=N)


def generate_message(t, Em, fm, wave_type):
    if wave_type == "Sine":
        return Em * np.sin(2 * np.pi * fm * t)
    elif wave_type == "Cosine":
        return Em * np.cos(2 * np.pi * fm * t)
    elif wave_type == "Triangular":
        # Sawtooth/triangular wave
        return Em * (2 * np.abs(2 * (t * fm - np.floor(t * fm + 0.5))) - 1)
    elif wave_type == "Two-Tone":
        return 0.7 * Em * np.sin(2 * np.pi * fm * t) + 0.3 * Em * np.sin(2 * np.pi * (2.5 * fm) * t)
    return Em * np.sin(2 * np.pi * fm * t)


# -----------------------------
# Main Title & Selection
# -----------------------------
st.title("Analog Pulse Modulation Visualizer")
st.markdown("Interactive tool for exploring **Pulse Amplitude Modulation (PAM)**, **Pulse Width Modulation (PWM)**, and **Pulse Position Modulation (PPM)**.")

modulation_type = st.selectbox("Select Modulation Type", ["PAM", "PWM", "PPM"], index=0)

# ==============================================================================
# 1. PULSE AMPLITUDE MODULATION (PAM)
# ==============================================================================
if modulation_type == "PAM":
    st.sidebar.header("1. Message Signal Variables")
    Em = st.sidebar.slider("Modulating Amplitude (Em) [V]", 0.1, 3.0, 1.0, 0.1)
    fm = st.sidebar.slider("Modulating Frequency (fm) [Hz]", 1, 20, 5, 1)
    wave_type = st.sidebar.selectbox("Message Waveform", ["Sine", "Cosine", "Triangular", "Two-Tone"], index=0)

    st.sidebar.header("2. Carrier Pulse Train")
    fs = st.sidebar.slider("Sampling Frequency (fs) [Hz]", 10, 200, 60, 5)
    Ac = st.sidebar.slider("Pulse Amplitude (Ac) [V]", 1.0, 5.0, 2.0, 0.2)
    duty_cycle = st.sidebar.slider("Duty Cycle (τ/Ts)", 0.05, 0.50, 0.25, 0.05)

    st.sidebar.header("3. PAM Specific Controls")
    pam_mode = st.sidebar.radio("Sampling Type", ["Flat-Top Sampling (Sample & Hold)", "Natural Sampling"], index=0)
    pam_carrier_type = st.sidebar.radio("Carrier Modulation Mode", ["Direct Carrier PAM [Ac + m(t)]", "Suppressed Carrier PAM [m(t) only]"], index=0)
    enable_eq = st.sidebar.checkbox("Apply Aperture Equalization on LPF Output", value=True)

    trace_visibility = init_visibility("pam", [
        "Message Signal m(t)",
        "Unmodulated Carrier c(t)",
        "PAM Waveform s(t)",
        "Sample Points",
        "Demodulated Output m̂(t)",
        "Equalized Output",
        "Aperture Sinc Envelope"
    ])

    st.sidebar.header("4. Trace Visibility")
    show_msg = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="pam_show_msg")
    show_car = st.sidebar.checkbox("Unmodulated Carrier", value=trace_visibility["Unmodulated Carrier c(t)"], key="pam_show_car")
    show_pam = st.sidebar.checkbox("PAM Waveform", value=trace_visibility["PAM Waveform s(t)"], key="pam_show_pam")
    show_pts = st.sidebar.checkbox("Sample Points", value=trace_visibility["Sample Points"], key="pam_show_pts")
    show_demod = st.sidebar.checkbox("Demodulated Output", value=trace_visibility["Demodulated Output m̂(t)"], key="pam_show_demod")
    show_eq = st.sidebar.checkbox("Equalized Output", value=trace_visibility["Equalized Output"], key="pam_show_eq")
    show_sinc = st.sidebar.checkbox("Aperture Sinc Envelope", value=trace_visibility["Aperture Sinc Envelope"], key="pam_show_sinc")

    trace_visibility.update({
        "Message Signal m(t)": show_msg,
        "Unmodulated Carrier c(t)": show_car,
        "PAM Waveform s(t)": show_pam,
        "Sample Points": show_pts,
        "Demodulated Output m̂(t)": show_demod,
        "Equalized Output": show_eq,
        "Aperture Sinc Envelope": show_sinc
    })

    # -----------------------------
    # Simulation Math - PAM
    # -----------------------------
    t_max = 0.4  # seconds
    num_samples = 4000
    t = np.linspace(0, t_max, num_samples)
    dt = t[1] - t[0]

    Ts = 1.0 / fs
    tau = duty_cycle * Ts

    m_t = generate_message(t, Em, fm, wave_type)

    # Direct Carrier Modulation: add message amplitude directly to carrier amplitude Ac
    is_direct_carrier = pam_carrier_type == "Direct Carrier PAM [Ac + m(t)]"
    baseline_A = Ac if is_direct_carrier else 0.0
    m_modulated = baseline_A + m_t

    # Generate Unmodulated Carrier Pulse Train c(t)
    c_t = np.zeros_like(t)
    # Generate PAM Waveform s(t)
    pam_t = np.zeros_like(t)

    sample_times = []
    sample_values = []
    pulse_heights = []

    # Calculate pulse locations
    n_pulses = int(np.ceil(t_max / Ts))
    for k in range(n_pulses):
        t_sample = k * Ts
        if t_sample <= t_max:
            sample_times.append(t_sample)
            val_m = generate_message(np.array([t_sample]), Em, fm, wave_type)[0]
            val_at_sample = baseline_A + val_m
            sample_values.append(val_m)
            pulse_heights.append(val_at_sample)

            pulse_mask = (t >= t_sample) & (t < t_sample + tau)
            c_t[pulse_mask] = Ac

            if pam_mode == "Flat-Top Sampling (Sample & Hold)":
                pam_t[pulse_mask] = val_at_sample
            else:  # Natural Sampling
                pam_t[pulse_mask] = m_modulated[pulse_mask]

    sample_times = np.array(sample_times)
    sample_values = np.array(sample_values)
    pulse_heights = np.array(pulse_heights)

    # Demodulation via LPF
    lpf_cutoff = 1.5 * fm
    demod_raw = apply_lpf(pam_t, 1.0 / dt, cutoff=lpf_cutoff)

    # Remove carrier DC baseline (Ac * duty_cycle) for Direct Carrier PAM
    if is_direct_carrier:
        demod_signal = (demod_raw - Ac * duty_cycle) / duty_cycle if duty_cycle > 0 else demod_raw
    else:
        demod_signal = demod_raw / duty_cycle if duty_cycle > 0 else demod_raw

    # Modulation index calculations
    mod_index = Em / Ac if Ac > 0 else 0.0
    is_overmodulated = is_direct_carrier and (Em > Ac)

    if is_overmodulated:
        st.warning(f"⚠️ **Overmodulation Warning:** Modulating Amplitude $E_m = {Em:.2f}$ V exceeds Carrier Amplitude $A_c = {Ac:.2f}$ V (Modulation Index $m_{{PAM}} = {mod_index*100:.1f}\\% > 100\\%$). Pulse height crosses zero, causing envelope distortion!")

    # Equalization calculation
    sinc_val = np.sinc(fm * tau)
    eq_gain = 1.0 / sinc_val if sinc_val != 0 else 1.0
    equalized_signal = demod_signal * eq_gain if pam_mode == "Flat-Top Sampling (Sample & Hold)" else demod_signal

    # FFT Spectrum
    N_fft = len(t)
    fft_freqs = np.fft.rfftfreq(N_fft, dt)
    fft_mags = np.abs(np.fft.rfft(pam_t)) * (2.0 / N_fft)

    # Theoretical sinc envelope for Flat-top PAM
    sinc_envelope = Ac * duty_cycle * np.abs(np.sinc(fft_freqs * tau))

    # Check Nyquist
    nyquist_freq = 2 * fm
    is_aliasing = fs < nyquist_freq

    if is_aliasing:
        st.error(f"⚠️ **Aliasing Warning:** Sampling Frequency $f_s = {fs}$ Hz is less than the Nyquist Rate $2f_m = {nyquist_freq}$ Hz! Signal reconstruction will suffer from spectral overlap distortion.")

    # -----------------------------
    # Plotting - PAM
    # -----------------------------
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time-Domain PAM Waveforms", "Frequency Domain Spectrum & Aperture Effect"),
        vertical_spacing=0.15
    )

    # Time Domain Traces
    fig.add_trace(go.Scatter(
        x=t, y=m_t, mode="lines", name="Message Signal m(t)",
        line=dict(color="purple", width=2, dash="dash"),
        visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=c_t, mode="lines", name="Unmodulated Carrier c(t)",
        line=dict(color="gray", width=1, dash="dot"),
        visible=True if trace_visibility["Unmodulated Carrier c(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=pam_t, mode="lines", name="PAM Waveform s(t)",
        line=dict(color="#1f77b4", width=2),
        visible=True if trace_visibility["PAM Waveform s(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sample_times, y=pulse_heights if is_direct_carrier else sample_values,
        mode="markers", name="Sample Points",
        marker=dict(color="red", size=7, symbol="diamond"),
        visible=True if trace_visibility["Sample Points"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=demod_signal, mode="lines", name="Demodulated Output m̂(t)",
        line=dict(color="orange", width=2),
        visible=True if trace_visibility["Demodulated Output m̂(t)"] else "legendonly"
    ), row=1, col=1)

    if pam_mode == "Flat-Top Sampling (Sample & Hold)":
        fig.add_trace(go.Scatter(
            x=t, y=equalized_signal, mode="lines", name="Equalized Output",
            line=dict(color="green", width=2, dash="dot"),
            visible=True if trace_visibility["Equalized Output"] else "legendonly"
        ), row=1, col=1)

    # Frequency Domain Traces
    max_freq_disp = min(4 * fs, 1.0 / dt / 2)
    freq_mask = fft_freqs <= max_freq_disp

    fig.add_trace(go.Scatter(
        x=fft_freqs[freq_mask], y=fft_mags[freq_mask], mode="lines", name="PAM Spectrum |S(f)|",
        line=dict(color="crimson", width=1.5),
        visible=True
    ), row=2, col=1)

    if trace_visibility["Aperture Sinc Envelope"] and pam_mode == "Flat-Top Sampling (Sample & Hold)":
        # Scale sinc envelope to match plot peak
        peak_mag = np.max(fft_mags[freq_mask]) if len(fft_mags[freq_mask]) > 0 else 1.0
        sinc_curve = peak_mag * np.abs(np.sinc(fft_freqs[freq_mask] * tau))
        fig.add_trace(go.Scatter(
            x=fft_freqs[freq_mask], y=sinc_curve, mode="lines", name="Aperture Sinc Envelope",
            line=dict(color="black", width=2, dash="dash"),
            visible=True
        ), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (V)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_freq_disp], row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (V)", row=2, col=1)
    fig.update_layout(height=720, showlegend=True, hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Performance Metrics - PAM
    # -----------------------------
    st.subheader("PAM Performance Metrics")
    metric_active = metric_buttons(["Modulation Index", "Nyquist & Aliasing", "Aperture Effect Loss", "Power & Duty Cycle", "Minimum Bandwidth"], "pam_active_metric")

    if metric_active:
        with st.container(border=True):
            if metric_active == "Modulation Index":
                st.markdown("### Modulation Index ($m_{PAM}$)")
                st.markdown(r"""
1. In **Direct Carrier PAM**, the message signal $m(t)$ is added directly to the unmodulated carrier pulse amplitude $A_c$:

$$
s_{PAM}(t) = [A_c + m(t)] \cdot c_0(t)
$$

2. The modulation index $m_{PAM}$ measures the ratio of peak modulating amplitude to carrier pulse amplitude:

$$
m_{PAM} = \frac{E_m}{A_c}
$$

3. Modulated pulse heights swing between $A_{max} = A_c + E_m$ and $A_{min} = A_c - E_m$.
""")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Carrier Amplitude (Ac)", f"{Ac:.2f} V")
                col2.metric("Message Amplitude (Em)", f"{Em:.2f} V")
                col3.metric("Modulation Index (m)", f"{mod_index*100:.1f} %")
                col4.metric("Pulse Height Swing", f"{Ac - Em:.2f} V – {Ac + Em:.2f} V")
                st.write(f"**Status:** {'⚠️ OVERMODULATED (m > 100%)' if is_overmodulated else '✅ NORMAL MODULATION (m ≤ 100%)'}")

            elif metric_active == "Nyquist & Aliasing":
                st.markdown("### Nyquist Criterion & Aliasing Analysis")
                st.markdown(r"""
1. The Nyquist Sampling Theorem states that to recover $m(t)$ without aliasing, $f_s \ge 2 f_m$.
2. **Nyquist Rate:** $f_{Nyq} = 2 f_m$
3. **Sampling Ratio:** $\frac{f_s}{f_m}$
""")
                col1, col2, col3 = st.columns(3)
                col1.metric("Modulating Freq (fm)", f"{fm} Hz")
                col2.metric("Nyquist Rate (2fm)", f"{nyquist_freq} Hz")
                col3.metric("Sampling Freq (fs)", f"{fs} Hz", delta=f"{fs - nyquist_freq} Hz vs Nyquist", delta_color="normal" if not is_aliasing else "inverse")
                st.write(f"**Sampling Ratio ($f_s/f_m$):** {fs/fm:.2f}")
                st.write(f"**Status:** {'❌ ALIASING PRESENT (Overlap of spectral components)' if is_aliasing else '✅ NO ALIASING (Perfect reconstruction possible)'}")

            elif metric_active == "Aperture Effect Loss":
                st.markdown("### Aperture Effect Loss (Flat-Top PAM)")
                st.markdown(r"""
1. In Flat-Top sampling, holding the sample value over width $\tau$ introduces high-frequency attenuation described by a sinc envelope:

$$
|H_{aperture}(f)| = \left| \frac{\sin(\pi f \tau)}{\pi f \tau} \right| = |\text{sinc}(f \tau)|
$$

2. At the modulating frequency $f = f_m$, the attenuation multiplier is $|H(f_m)|$.
3. To compensate, an equalizer with gain $H_{eq}(f) = \frac{1}{\text{sinc}(f \tau)}$ is placed after the Low-Pass Filter.
""")
                loss_db = 20 * np.log10(sinc_val) if sinc_val > 0 else -99
                col1, col2, col3 = st.columns(3)
                col1.metric("Pulse Width (τ)", f"{tau*1e3:.2f} ms")
                col2.metric("Aperture Factor |H(fm)|", f"{sinc_val:.4f}")
                col3.metric("Attenuation at fm", f"{loss_db:.2f} dB")
                st.write(f"**Required Equalizer Gain at $f_m$:** {eq_gain:.4f} ({20*np.log10(eq_gain):.2f} dB)")

            elif metric_active == "Power & Duty Cycle":
                st.markdown("### Power & Transmission Efficiency")
                st.markdown(r"""
1. **Duty Cycle ($D$):** Ratio of pulse duration $\tau$ to sampling period $T_s$:

$$
D = \frac{\tau}{T_s} = \tau f_s
$$

2. **Unmodulated Carrier Power ($P_c$):**

$$
P_c = D \cdot A_c^2
$$

3. **Total Average Transmitted Power ($P_{PAM}$):**
For Direct Carrier PAM, total average transmitted power across a $1\Omega$ load is:

$$
P_{PAM} = D \cdot \left( A_c^2 + \frac{E_m^2}{2} \right) = P_c \left( 1 + \frac{m_{PAM}^2}{2} \right)
$$
""")
                p_carrier = duty_cycle * (Ac**2)
                p_sideband = duty_cycle * (Em**2 / 2.0)
                p_total = p_carrier + p_sideband if is_direct_carrier else p_sideband
                eff = (p_sideband / p_total) if p_total > 0 else 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Carrier Power (Pc)", f"{p_carrier:.3f} W")
                col2.metric("Sideband Power (Psb)", f"{p_sideband:.3f} W")
                col3.metric("Total Power (Ppam)", f"{p_total:.3f} W")
                col4.metric("Efficiency (η)", f"{eff*100:.1f} %")

            elif metric_active == "Minimum Bandwidth":
                st.markdown("### Minimum Transmission Bandwidth")
                st.markdown(r"""
1. The minimum channel bandwidth $BW_{min}$ for pulse transmission depends on the pulse width $\tau$:

$$
BW_{min} \approx \frac{1}{2\tau}
$$

2. The main lobe bandwidth of the sinc spectrum extends to $\frac{1}{\tau}$.
""")
                bw_min = 1.0 / (2 * tau)
                bw_mainlobe = 1.0 / tau
                col1, col2 = st.columns(2)
                col1.metric("Min Bandwidth (1/2τ)", f"{bw_min:.1f} Hz")
                col2.metric("Main-Lobe Width (1/τ)", f"{bw_mainlobe:.1f} Hz")

    # -----------------------------
    # Theory & Derivations - PAM
    # -----------------------------
    st.markdown("## About Pulse Amplitude Modulation (PAM)")
    st.markdown(r"""
- **Pulse Amplitude Modulation (PAM)** is an analog pulse modulation technique where the **amplitude** of a sequence of regularly spaced rectangular pulses is varied in proportion to the sample values of the continuous message signal $m(t)$.
- **Direct Carrier Modulation:** The message signal $m(t)$ is added directly to the unmodulated carrier pulse peak amplitude $A_c$, producing pulse heights $A(t) = A_c + m(t)$ (Natural) or $A_k = A_c + m(k T_s)$ (Flat-top). The modulation index is $m_{PAM} = E_m / A_c$.
- **Natural Sampling:** The pulse tops follow the exact shape of $[A_c + m(t)]$ during the pulse duration $\tau$.
- **Flat-Top Sampling:** The pulse top is held constant at $[A_c + m(k T_s)]$ for the duration $\tau$ of the pulse using a Sample-and-Hold circuit.
- **Aperture Effect:** Flat-top sampling causes high-frequency attenuation described by $H(f) = \tau \text{sinc}(f \tau) e^{-j \pi f \tau}$, which is compensated using an equalizer $H_{eq}(f) = 1/\text{sinc}(f \tau)$.
- **Demodulation:** LPF filtering of Direct Carrier PAM yields $D \cdot [A_c + m(t)]$. Subtracting the carrier baseline $D \cdot A_c$ and scaling by $1/D$ recovers $m(t)$.
""")

    st.subheader("PAM Derivations & Mathematical Foundations")
    derivation_active = derivation_buttons("pam_active_derivation", ["Direct Carrier PAM Math", "Flat-Top Sampling & Aperture Effect", "Equalization Filter", "Bandwidth & Power"])

    if derivation_active:
        with st.container(border=True):
            if derivation_active == "Direct Carrier PAM Math":
                st.markdown("### Direct Carrier PAM Mathematical Derivation")
                st.markdown(r"""
**1. Unmodulated Unit Pulse Train:**
A periodic train of unit amplitude pulses $c_0(t)$ with period $T_s$ and width $\tau$ has Fourier series:

$$
c_0(t) = \sum_{n=-\infty}^{\infty} C_n e^{j n \omega_s t}, \quad C_n = \frac{\tau}{T_s} \text{sinc}(n f_s \tau) = D \text{sinc}(n D)
$$

**2. Direct Carrier PAM Expression:**
Adding the message signal $m(t)$ directly to the carrier pulse amplitude $A_c$:

$$
s_{PAM}(t) = [A_c + m(t)] \cdot c_0(t) = [A_c + m(t)] \sum_{n=-\infty}^{\infty} C_n e^{j n \omega_s t}
$$

Distributing terms yields:

$$
s_{PAM}(t) = A_c c_0(t) + D m(t) + \sum_{n=1}^{\infty} 2 C_n m(t) \cos(n \omega_s t)
$$

**3. Demodulation via Low-Pass Filter:**
Passing $s_{PAM}(t)$ through an LPF with cutoff frequency $f_m < f_{cut} < f_s - f_m$ isolates the DC and baseband components:

$$
v_{LPF}(t) = A_c D + D m(t) = D [A_c + m(t)]
$$

Subtracting the constant DC voltage $V_{DC} = A_c D$ and dividing by $D$ yields the exact message signal $m(t)$!
""")

            elif derivation_active == "Flat-Top Sampling & Aperture Effect":
                st.markdown("### Flat-Top Sampling & Aperture Effect Derivation")
                st.markdown(r"""
**1. Mathematical Model of Flat-Top Sampling:**
Flat-top sampling is obtained by convolving an instantaneously sampled signal $m_\delta(t)$ with a rectangular pulse $p(t)$ of duration $\tau$:

$$
m_\delta(t) = m(t) \cdot \delta_{T_s}(t) = \sum_{k=-\infty}^{\infty} m(k T_s) \delta(t - k T_s)
$$

$$
s_{flat}(t) = m_\delta(t) * p(t) = \sum_{k=-\infty}^{\infty} m(k T_s) p(t - k T_s)
$$

where $p(t) = 1$ for $0 \le t \le \tau$ and $0$ elsewhere.

**2. Fourier Transform & Aperture Function:**
By the Convolution Theorem, multiplication in frequency domain gives:

$$
S_{flat}(f) = \mathcal{F}\{m_\delta(t)\} \cdot P(f) = \left[ f_s \sum_{n=-\infty}^{\infty} M(f - n f_s) \right] \cdot P(f)
$$

The Fourier transform of the rectangular pulse $p(t)$ of width $\tau$ is:

$$
P(f) = \tau \text{sinc}(f \tau) e^{-j \pi f \tau}
$$

**3. Aperture Distortion:**
Because $P(f)$ contains the factor $\text{sinc}(f \tau) = \frac{\sin(\pi f \tau)}{\pi f \tau}$, higher frequency components of $M(f)$ are attenuated relative to lower frequencies. This roll-off effect is called the **Aperture Effect**.
""")

            elif derivation_active == "Equalization Filter":
                st.markdown("### Equalization Filter Derivation")
                st.markdown(r"""
**1. Purpose:**
To eliminate aperture distortion in Flat-Top PAM, an **equalizer filter** is placed in cascade with the Low-Pass Filter (LPF).

**2. Transfer Function $H_{eq}(f)$:**
The combined transfer function of the sampling pulse shape and the equalizer should yield a flat frequency response over the message bandwidth $|f| \le f_m$:

$$
H_{net}(f) = P(f) \cdot H_{eq}(f) = K \quad \text{(Constant)}
$$

Since $P(f) = \tau \text{sinc}(f \tau) e^{-j \pi f \tau}$, the equalizer transfer function magnitude must be:

$$
|H_{eq}(f)| = \frac{1}{|P(f)|} = \frac{1}{\tau \text{sinc}(f \tau)} = \frac{\pi f \tau}{\tau \sin(\pi f \tau)} = \frac{\pi f}{\sin(\pi f \tau)}
$$

**3. Equalized Signal:**
Multiplying the LPF output spectrum by $H_{eq}(f)$ completely restores the flat spectrum of the original message signal $m(t)$.
""")

            elif derivation_active == "Bandwidth & Power":
                st.markdown("### PAM Bandwidth & Power Derivation")
                st.markdown(r"""
**1. Transmission Bandwidth:**
- The theoretical bandwidth of a rectangular pulse of duration $\tau$ is infinite due to sinc sidelobes.
- Practically, the transmission channel must pass at least the main lobe of the sinc function, which extends up to $f = \frac{1}{\tau}$.
- To prevent pulse overlap and preserve pulse amplitude:

$$
BW_{min} \approx \frac{1}{2\tau}
$$

Since $\tau \ll T_s$, the required bandwidth for PAM is significantly larger than the message bandwidth $f_m$.

**2. Power Calculation:**
- Peak power of a pulse: $P_{peak} = \frac{A_c^2}{R}$ (assuming $R=1\Omega$).
- Total average power over one period $T_s$:

$$
P_{avg} = \frac{1}{T_s} \int_{0}^{\tau} s_{PAM}^2(t) dt = \frac{\tau}{T_s} A_c^2 = D \cdot A_c^2
$$

where $D = \tau/T_s$ is the duty cycle.
""")

# ==============================================================================
# 2. PULSE WIDTH MODULATION (PWM)
# ==============================================================================
elif modulation_type == "PWM":
    st.sidebar.header("1. Message Signal Variables")
    Em = st.sidebar.slider("Modulating Amplitude (Em) [V]", 0.1, 3.0, 1.0, 0.1)
    fm = st.sidebar.slider("Modulating Frequency (fm) [Hz]", 1, 20, 5, 1)
    wave_type = st.sidebar.selectbox("Message Waveform", ["Sine", "Cosine", "Triangular", "Two-Tone"], index=0)

    st.sidebar.header("2. Carrier & Ramp Wave")
    fs = st.sidebar.slider("Sampling Frequency (fs) [Hz]", 10, 200, 60, 5)
    Ac = st.sidebar.slider("Pulse Amplitude (Ac) [V]", 1.0, 5.0, 2.0, 0.2)
    tau0_ratio = st.sidebar.slider("Unmodulated Duty Cycle (τ₀/Ts)", 0.1, 0.8, 0.5, 0.05)

    st.sidebar.header("3. PWM Specific Controls")
    pwm_mode = st.sidebar.selectbox("Pulse Alignment", ["Trailing Edge (Fixed Start)", "Leading Edge (Fixed End)", "Center Aligned (Symmetrical)"], index=0)
    k_pwm = st.sidebar.slider("Modulation Sensitivity (k_pwm)", 0.1, 0.9, 0.4, 0.05)
    carrier_shape = st.sidebar.radio("Comparator Ramp Type", ["Sawtooth Ramp", "Triangular Wave"], index=0)

    trace_visibility = init_visibility("pwm", [
        "Message Signal m(t)",
        "Comparator Reference Ramp v_ref(t)",
        "PWM Waveform s(t)",
        "Pulse Width Envelope τ(t)",
        "Demodulated Output m̂(t)"
    ])

    st.sidebar.header("4. Trace Visibility")
    show_msg = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="pwm_show_msg")
    show_ramp = st.sidebar.checkbox("Comparator Reference Ramp", value=trace_visibility["Comparator Reference Ramp v_ref(t)"], key="pwm_show_ramp")
    show_pwm = st.sidebar.checkbox("PWM Waveform", value=trace_visibility["PWM Waveform s(t)"], key="pwm_show_pwm")
    show_env = st.sidebar.checkbox("Pulse Width Envelope", value=trace_visibility["Pulse Width Envelope τ(t)"], key="pwm_show_env")
    show_demod = st.sidebar.checkbox("Demodulated Output", value=trace_visibility["Demodulated Output m̂(t)"], key="pwm_show_demod")

    trace_visibility.update({
        "Message Signal m(t)": show_msg,
        "Comparator Reference Ramp v_ref(t)": show_ramp,
        "PWM Waveform s(t)": show_pwm,
        "Pulse Width Envelope τ(t)": show_env,
        "Demodulated Output m̂(t)": show_demod
    })

    # -----------------------------
    # Simulation Math - PWM
    # -----------------------------
    t_max = 0.4
    num_samples = 4000
    t = np.linspace(0, t_max, num_samples)
    dt = t[1] - t[0]

    Ts = 1.0 / fs
    tau0 = tau0_ratio * Ts

    m_t = generate_message(t, Em, fm, wave_type)
    # Normalized message between -1 and +1
    m_norm = m_t / Em

    # Generate Comparator Ramp / Reference Wave
    v_ref = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = (ti % Ts) / Ts
        if carrier_shape == "Sawtooth Ramp":
            v_ref[i] = (2 * phase - 1) * Em
        else:  # Triangular Wave
            v_ref[i] = (1 - 4 * np.abs(phase - 0.5)) * Em

    # Generate PWM Waveform s(t)
    pwm_t = np.zeros_like(t)
    tau_envelope = np.zeros_like(t)

    n_pulses = int(np.ceil(t_max / Ts))
    tau_min = Ts
    tau_max = 0.0

    for k in range(n_pulses):
        t_start_period = k * Ts
        t_sample = t_start_period  # Sample at start of period
        if t_sample <= t_max:
            val_m = generate_message(np.array([t_sample]), Em, fm, wave_type)[0]
            val_norm = val_m / Em

            # Width variation proportional to message sample
            delta_tau = k_pwm * val_norm * (0.45 * Ts)
            tau_k = np.clip(tau0 + delta_tau, 0.02 * Ts, 0.98 * Ts)

            tau_min = min(tau_min, tau_k)
            tau_max = max(tau_max, tau_k)

            # Determine pulse start and end based on alignment
            if pwm_mode == "Trailing Edge (Fixed Start)":
                p_start = t_start_period
                p_end = t_start_period + tau_k
            elif pwm_mode == "Leading Edge (Fixed End)":
                p_start = (k + 1) * Ts - tau_k
                p_end = (k + 1) * Ts
            else:  # Center Aligned
                p_start = t_start_period + (Ts - tau_k) / 2.0
                p_end = t_start_period + (Ts + tau_k) / 2.0

            pulse_mask = (t >= p_start) & (t < p_end)
            pwm_t[pulse_mask] = Ac

            period_mask = (t >= t_start_period) & (t < (k + 1) * Ts)
            tau_envelope[period_mask] = tau_k

    # Demodulation via LPF (PWM average value is directly proportional to pulse width!)
    lpf_cutoff = 1.5 * fm
    demod_raw = apply_lpf(pwm_t, 1.0 / dt, cutoff=lpf_cutoff)

    # Scale and center demodulated signal
    avg_dc = Ac * (tau0 / Ts)
    demod_signal = (demod_raw - avg_dc) * (Em / (Ac * k_pwm * 0.45))

    # FFT Spectrum
    N_fft = len(t)
    fft_freqs = np.fft.rfftfreq(N_fft, dt)
    fft_mags = np.abs(np.fft.rfft(pwm_t)) * (2.0 / N_fft)

    # -----------------------------
    # Plotting - PWM
    # -----------------------------
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time-Domain PWM Waveforms & Comparator Ramp", "Frequency Domain Spectrum (Intermodulation Sidebands)"),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(
        x=t, y=m_t, mode="lines", name="Message Signal m(t)",
        line=dict(color="purple", width=2, dash="dash"),
        visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=v_ref, mode="lines", name="Comparator Reference Ramp v_ref(t)",
        line=dict(color="gray", width=1.2, dash="dot"),
        visible=True if trace_visibility["Comparator Reference Ramp v_ref(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=pwm_t, mode="lines", name="PWM Waveform s(t)",
        line=dict(color="#2ca02c", width=2),
        visible=True if trace_visibility["PWM Waveform s(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=tau_envelope * 1e3, mode="lines", name="Pulse Width Envelope τ(t) [ms]",
        line=dict(color="blue", width=1.5, dash="dashdot"),
        visible=True if trace_visibility["Pulse Width Envelope τ(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=demod_signal, mode="lines", name="Demodulated Output m̂(t)",
        line=dict(color="orange", width=2),
        visible=True if trace_visibility["Demodulated Output m̂(t)"] else "legendonly"
    ), row=1, col=1)

    max_freq_disp = min(4 * fs, 1.0 / dt / 2)
    freq_mask = fft_freqs <= max_freq_disp

    fig.add_trace(go.Scatter(
        x=fft_freqs[freq_mask], y=fft_mags[freq_mask], mode="lines", name="PWM Spectrum |S(f)|",
        line=dict(color="forestgreen", width=1.5),
        visible=True
    ), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (V)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_freq_disp], row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (V)", row=2, col=1)
    fig.update_layout(height=720, showlegend=True, hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Performance Metrics - PWM
    # -----------------------------
    st.subheader("PWM Performance Metrics")
    metric_active = metric_buttons(["Pulse Width Limits", "Modulation Index", "Average Power Variation", "Occupied Bandwidth"], "pwm_active_metric")

    if metric_active:
        with st.container(border=True):
            if metric_active == "Pulse Width Limits":
                st.markdown("### Pulse Width Limits & Duty Cycle Range")
                st.markdown(r"""
1. The pulse width varies between $\tau_{min}$ and $\tau_{max}$ depending on message amplitude $E_m$:

$$
\tau_{min} = \tau_0 - \Delta \tau_{max}, \quad \tau_{max} = \tau_0 + \Delta \tau_{max}
$$

2. Overlap distortion occurs if $\tau_{max} \ge T_s$ or $\tau_{min} \le 0$.
""")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nominal Width (τ₀)", f"{tau0*1e3:.2f} ms")
                col2.metric("Min Width (τmin)", f"{tau_min*1e3:.2f} ms")
                col3.metric("Max Width (τmax)", f"{tau_max*1e3:.2f} ms")
                col4.metric("Duty Cycle Span", f"{(tau_min/Ts)*100:.0f}% – {(tau_max/Ts)*100:.0f}%")

            elif metric_active == "Modulation Index":
                st.markdown("### PWM Modulation Index")
                st.markdown(r"""
1. The PWM modulation index $\beta_{PWM}$ represents the relative pulse width excursion:

$$
\beta_{PWM} = \frac{\Delta \tau_{max}}{\tau_0} = \frac{\tau_{max} - \tau_{min}}{2 \tau_0}
$$
""")
                beta_pwm = (tau_max - tau_min) / (2 * tau0) if tau0 > 0 else 0.0
                col1, col2 = st.columns(2)
                col1.metric("Max Shift (Δτmax)", f"{(tau_max - tau0)*1e3:.2f} ms")
                col2.metric("PWM Modulation Index (β_pwm)", f"{beta_pwm:.3f}")

            elif metric_active == "Average Power Variation":
                st.markdown("### Average Power Variation")
                st.markdown(r"""
1. Unlike PPM, PWM transmitted power varies directly with the message signal amplitude!
2. Instantaneous duty cycle $D(t) = \frac{\tau(t)}{T_s}$.
3. Average Power:

$$
P_{avg} = A_c^2 \cdot \bar{D}
$$
""")
                avg_duty = np.mean(tau_envelope) / Ts
                p_avg = (Ac**2) * avg_duty
                col1, col2, col3 = st.columns(3)
                col1.metric("Pulse Amplitude (Ac)", f"{Ac} V")
                col2.metric("Mean Duty Cycle (D_bar)", f"{avg_duty*100:.1f} %")
                col3.metric("Average Power (Pavg)", f"{p_avg:.3f} W")

            elif metric_active == "Occupied Bandwidth":
                st.markdown("### PWM Bandwidth Estimation")
                st.markdown(r"""
1. The minimum transmission bandwidth is governed by the narrowest pulse duration $\tau_{min}$:

$$
BW_{PWM} \approx \frac{1}{2 \tau_{min}}
$$
""")
                bw_pwm = 1.0 / (2 * tau_min) if tau_min > 0 else 0.0
                col1, col2 = st.columns(2)
                col1.metric("Narrowest Pulse (τmin)", f"{tau_min*1e3:.2f} ms")
                col2.metric("Occupied Bandwidth", f"{bw_pwm:.1f} Hz")

    # -----------------------------
    # Theory & Derivations - PWM
    # -----------------------------
    st.markdown("## About Pulse Width Modulation (PWM)")
    st.markdown(r"""
- **Pulse Width Modulation (PWM)** (also known as Pulse Duration Modulation, PDM) varies the **duration or width** of pulses proportional to the sample values of the modulating signal $m(t)$, while keeping pulse amplitude and sampling frequency constant.
- **Generation:** PWM waves are easily generated by comparing the message signal $m(t)$ with a periodic ramp/sawtooth reference wave using a high-speed comparator.
- **Edge Alignments:**
  - **Trailing Edge:** The leading edge of each pulse is fixed at $t = k T_s$, while the trailing edge varies with $m(t)$.
  - **Leading Edge:** The trailing edge of each pulse is fixed at $t = (k+1) T_s$, while the leading edge varies.
  - **Center Aligned:** Both edges vary symmetrically around the center of the sampling period.
- **Demodulation:** Low-pass filtering directly recovers $m(t)$ because the average value of a PWM pulse over a period $T_s$ is $\bar{V} = A_c \frac{\tau_k}{T_s} \propto m(k T_s)$.
- **Applications:** Motor speed control, switching power supplies (class-D amplifiers, DC-DC converters), and power electronics.
""")

    st.subheader("PWM Derivations & Mathematical Foundations")
    derivation_active = derivation_buttons("pwm_active_derivation", ["Comparator Generation Mechanics", "Pulse Width Derivation", "LPF Demodulation Principle", "Power & Bandwidth Analysis"])

    if derivation_active:
        with st.container(border=True):
            if derivation_active == "Comparator Generation Mechanics":
                st.markdown("### PWM Generation via Comparator Derivation")
                st.markdown(r"""
**1. Comparator Setup:**
Consider a comparator with inputs:
- Non-inverting input: Message signal $m(t)$
- Inverting input: Periodic sawtooth ramp $v_{saw}(t)$ with period $T_s$ and slope $E_{max}/T_s$.

**2. Output Logic:**

$$
s_{PWM}(t) = \begin{cases} A_c & \text{if } m(t) > v_{saw}(t) \\ 0 & \text{if } m(t) \le v_{saw}(t) \end{cases}
$$

**3. Trailing-Edge Pulse Duration:**
The ramp equation in period $k$ is $v_{saw}(t) = -E_{max} + \frac{2 E_{max}}{T_s} (t - k T_s)$.
The pulse stays HIGH from $t = k T_s$ until $v_{saw}(t)$ crosses $m(k T_s)$:

$$
m(k T_s) = -E_{max} + \frac{2 E_{max}}{T_s} \tau_k \implies \tau_k = \frac{T_s}{2} \left( 1 + \frac{m(k T_s)}{E_{max}} \right)
$$

This proves that pulse width $\tau_k$ varies linearly with message sample $m(k T_s)$!
""")

            elif derivation_active == "Pulse Width Derivation":
                st.markdown("### Mathematical Expression for PWM Waveform")
                st.markdown(r"""
**1. Time-Domain Expression:**
Let the unmodulated pulse width be $\tau_0 = D_0 T_s$. The modulated pulse duration is:

$$
\tau(t) = \tau_0 + k_{PWM} \cdot m(t)
$$

**2. Fourier Series of Trailing-Edge PWM:**
Expanding $s_{PWM}(t)$ in a double Fourier series yields:

$$
s_{PWM}(t) = A_c D_0 + \frac{A_c k_{PWM}}{T_s} m(t) + \sum_{m=1}^{\infty} \sum_{n=-\infty}^{\infty} C_{mn} \cos(m \omega_s t + n \omega_m t)
$$

**3. Spectrum Takeaway:**
The spectrum contains:
- DC component: $A_c D_0$
- Baseband message signal: $\frac{A_c k_{PWM}}{T_s} m(t)$
- Harmonics of sampling frequency $m f_s$ with intermodulation sidebands $m f_s \pm n f_m$.
""")

            elif derivation_active == "LPF Demodulation Principle":
                st.markdown("### LPF Demodulation Principle Derivation")
                st.markdown(r"""
**1. Average Voltage in $k$-th Period:**
The average voltage of the PWM signal over one sampling interval $T_s$ is:

$$
\bar{v}_k = \frac{1}{T_s} \int_{k T_s}^{(k+1)T_s} s_{PWM}(t) dt = \frac{1}{T_s} (A_c \cdot \tau_k) = A_c \frac{\tau_k}{T_s}
$$

**2. Substituting $\tau_k$:**
Substitute $\tau_k = \tau_0 + k_{PWM} m(k T_s)$:

$$
\bar{v}_k = A_c \frac{\tau_0}{T_s} + \frac{A_c k_{PWM}}{T_s} m(k T_s)
$$

**3. Low-Pass Filter Output:**
Passing $s_{PWM}(t)$ through an LPF with cutoff frequency $f_m < f_{cut} < f_s - f_m$ removes high-frequency harmonics, yielding:

$$
v_{out}(t) = V_{DC} + K \cdot m(t)
$$

Blocking the DC offset $V_{DC} = A_c \frac{\tau_0}{T_s}$ directly recovers the exact message $m(t)$!
""")

            elif derivation_active == "Power & Bandwidth Analysis":
                st.markdown("### PWM Power & Bandwidth Derivation")
                st.markdown(r"""
**1. Power Variation:**
Because pulse width varies with the message signal, the duty cycle $D(t) = \tau(t)/T_s$ varies dynamically.
The average power delivered over a complete message period $T_m = 1/f_m$ is:

$$
P_{avg} = \frac{1}{T_m} \int_0^{T_m} s_{PWM}^2(t) dt = A_c^2 \bar{D} = A_c^2 \frac{\tau_0}{T_s}
$$

For a zero-mean message signal $m(t)$, $\bar{D} = D_0 = \tau_0/T_s$.

**2. Bandwidth Requirement:**
The minimum bandwidth required to transmit a pulse of minimum width $\tau_{min}$ without excessive distortion is:

$$
BW_{min} \approx \frac{1}{2 \tau_{min}}
$$
""")

# ==============================================================================
# 3. PULSE POSITION MODULATION (PPM)
# ==============================================================================
elif modulation_type == "PPM":
    st.sidebar.header("1. Message Signal Variables")
    Em = st.sidebar.slider("Modulating Amplitude (Em) [V]", 0.1, 3.0, 1.0, 0.1)
    fm = st.sidebar.slider("Modulating Frequency (fm) [Hz]", 1, 20, 5, 1)
    wave_type = st.sidebar.selectbox("Message Waveform", ["Sine", "Cosine", "Triangular", "Two-Tone"], index=0)

    st.sidebar.header("2. Carrier & Pulse Parameters")
    fs = st.sidebar.slider("Sampling Frequency (fs) [Hz]", 10, 200, 60, 5)
    Ac = st.sidebar.slider("Pulse Amplitude (Ac) [V]", 1.0, 5.0, 2.0, 0.2)
    tau_ppm_ratio = st.sidebar.slider("Fixed Pulse Width (τ/Ts)", 0.02, 0.20, 0.08, 0.01)

    st.sidebar.header("3. PPM Specific Controls")
    k_ppm = st.sidebar.slider("Position Sensitivity (k_ppm)", 0.05, 0.40, 0.25, 0.05)

    trace_visibility = init_visibility("ppm", [
        "Message Signal m(t)",
        "Unmodulated Carrier c(t)",
        "Reference Clock Pulse Train c_clk(t)",
        "Intermediate PWM Trailing Edge",
        "PPM Waveform s(t)",
        "Regenerated PWM Waveform",
        "Demodulated Output m̂(t)"
    ])

    st.sidebar.header("4. Trace Visibility")
    show_msg = st.sidebar.checkbox("Message Signal", value=trace_visibility["Message Signal m(t)"], key="ppm_show_msg")
    show_car = st.sidebar.checkbox("Unmodulated Carrier", value=trace_visibility["Unmodulated Carrier c(t)"], key="ppm_show_car")
    show_clk = st.sidebar.checkbox("Reference Clock Pulse Train", value=trace_visibility["Reference Clock Pulse Train c_clk(t)"], key="ppm_show_clk")
    show_pwm_ref = st.sidebar.checkbox("Intermediate PWM Trailing Edge", value=trace_visibility["Intermediate PWM Trailing Edge"], key="ppm_show_pwm_ref")
    show_ppm = st.sidebar.checkbox("PPM Waveform", value=trace_visibility["PPM Waveform s(t)"], key="ppm_show_ppm")
    show_regen = st.sidebar.checkbox("Regenerated PWM Waveform", value=trace_visibility["Regenerated PWM Waveform"], key="ppm_show_regen")
    show_demod = st.sidebar.checkbox("Demodulated Output", value=trace_visibility["Demodulated Output m̂(t)"], key="ppm_show_demod")

    trace_visibility.update({
        "Message Signal m(t)": show_msg,
        "Unmodulated Carrier c(t)": show_car,
        "Reference Clock Pulse Train c_clk(t)": show_clk,
        "Intermediate PWM Trailing Edge": show_pwm_ref,
        "PPM Waveform s(t)": show_ppm,
        "Regenerated PWM Waveform": show_regen,
        "Demodulated Output m̂(t)": show_demod
    })

    # -----------------------------
    # Simulation Math - PPM
    # -----------------------------
    t_max = 0.4
    num_samples = 4000
    t = np.linspace(0, t_max, num_samples)
    dt = t[1] - t[0]

    Ts = 1.0 / fs
    tau_ppm = tau_ppm_ratio * Ts
    nominal_delay = 0.45 * Ts

    m_t = generate_message(t, Em, fm, wave_type)

    # Unmodulated Carrier Pulse Train (pulses of width tau_ppm at nominal position k*Ts + nominal_delay)
    c_carrier = np.zeros_like(t)
    # Reference Clock Pulse Train (pulses of width tau_ppm at period start k*Ts)
    c_clk = np.zeros_like(t)
    # Intermediate PWM Trailing Edge
    pwm_edge_t = np.zeros_like(t)
    # PPM Waveform
    ppm_t = np.zeros_like(t)
    # Regenerated PWM Waveform (SR Flip-Flop: Set at Clock, Reset at PPM)
    regen_pwm = np.zeros_like(t)

    n_pulses = int(np.ceil(t_max / Ts))
    has_pulse_overlap = False

    for k in range(n_pulses):
        t_clk = k * Ts
        if t_clk <= t_max:
            # Reference Clock pulse at k * Ts of width tau_ppm
            clk_mask = (t >= t_clk) & (t < t_clk + tau_ppm)
            c_clk[clk_mask] = Ac

            # Unmodulated Carrier pulse at nominal delay (k * Ts + nominal_delay) of width tau_ppm
            t_unmod_start = t_clk + nominal_delay
            t_unmod_end = t_unmod_start + tau_ppm
            car_mask = (t >= t_unmod_start) & (t < t_unmod_end)
            c_carrier[car_mask] = Ac

            val_m = generate_message(np.array([t_clk]), Em, fm, wave_type)[0]
            val_norm = val_m / Em

            # Position shift proportional to message value
            shift = nominal_delay + k_ppm * val_norm * Ts
            t_ppm_start = t_clk + shift
            t_ppm_end = t_ppm_start + tau_ppm

            # Pulse overlap check with next clock period (k+1)*Ts
            if t_ppm_end >= (k + 1) * Ts:
                has_pulse_overlap = True

            # Modulated PPM pulse
            ppm_mask = (t >= t_ppm_start) & (t < t_ppm_end)
            ppm_t[ppm_mask] = Ac

            # Intermediate PWM trailing edge line for visualization
            pwm_mask = (t >= t_clk) & (t < t_ppm_start)
            pwm_edge_t[pwm_mask] = Ac * 0.5

            # SR Flip-Flop Regeneration (Set at t_clk, Reset at t_ppm_start)
            regen_pwm[pwm_mask] = Ac

    if has_pulse_overlap:
        st.error("⚠️ **Pulse Overlap / Collision Alert:** High position sensitivity $k_{ppm}$ causes PPM pulses to overlap into the next sampling period! Reduce $k_{ppm}$ or increase sampling frequency.")

    # Demodulation: Pass regenerated PWM through LPF!
    lpf_cutoff = 1.5 * fm
    demod_raw = apply_lpf(regen_pwm, 1.0 / dt, cutoff=lpf_cutoff)
    avg_dc = Ac * (nominal_delay / Ts)
    demod_signal = (demod_raw - avg_dc) * (Em / (Ac * k_ppm))

    # FFT Spectrum
    N_fft = len(t)
    fft_freqs = np.fft.rfftfreq(N_fft, dt)
    fft_mags = np.abs(np.fft.rfft(ppm_t)) * (2.0 / N_fft)

    # -----------------------------
    # Plotting - PPM
    # -----------------------------
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time-Domain PPM Waveforms & Flip-Flop Regeneration", "Frequency Domain Spectrum (Carrier Harmonics & Phase Sidebands)"),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(
        x=t, y=m_t, mode="lines", name="Message Signal m(t)",
        line=dict(color="purple", width=2, dash="dash"),
        visible=True if trace_visibility["Message Signal m(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=c_carrier, mode="lines", name="Unmodulated Carrier c(t)",
        line=dict(color="gray", width=1.2, dash="dash"),
        visible=True if trace_visibility["Unmodulated Carrier c(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=c_clk, mode="lines", name="Reference Clock c_clk(t)",
        line=dict(color="darkgray", width=1, dash="dot"),
        visible=True if trace_visibility["Reference Clock Pulse Train c_clk(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=pwm_edge_t, mode="lines", name="Intermediate PWM Trailing Edge",
        line=dict(color="lightblue", width=1, dash="dash"),
        visible=True if trace_visibility["Intermediate PWM Trailing Edge"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=ppm_t, mode="lines", name="PPM Waveform s(t)",
        line=dict(color="crimson", width=2),
        visible=True if trace_visibility["PPM Waveform s(t)"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=regen_pwm, mode="lines", name="Regenerated PWM Waveform",
        line=dict(color="teal", width=1.5, dash="dot"),
        visible=True if trace_visibility["Regenerated PWM Waveform"] else "legendonly"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=demod_signal, mode="lines", name="Demodulated Output m̂(t)",
        line=dict(color="orange", width=2),
        visible=True if trace_visibility["Demodulated Output m̂(t)"] else "legendonly"
    ), row=1, col=1)

    max_freq_disp = min(4 * fs, 1.0 / dt / 2)
    freq_mask = fft_freqs <= max_freq_disp

    fig.add_trace(go.Scatter(
        x=fft_freqs[freq_mask], y=fft_mags[freq_mask], mode="lines", name="PPM Spectrum |S(f)|",
        line=dict(color="firebrick", width=1.5),
        visible=True
    ), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (V)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, max_freq_disp], row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (V)", row=2, col=1)
    fig.update_layout(height=720, showlegend=True, hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Performance Metrics - PPM
    # -----------------------------
    st.subheader("PPM Performance Metrics")
    metric_active = metric_buttons(["Max Time Displacement", "Guard Interval & Collision", "Constant Power Property", "Noise Immunity & Jitter"], "ppm_active_metric")

    if metric_active:
        with st.container(border=True):
            if metric_active == "Max Time Displacement":
                st.markdown("### Maximum Time Displacement")
                st.markdown(r"""
1. Pulse position varies relative to nominal position $\tau_0$:

$$
\Delta t_{max} = k_{PPM} \cdot T_s
$$
""")
                dt_max = k_ppm * Ts
                col1, col2, col3 = st.columns(3)
                col1.metric("Sampling Period (Ts)", f"{Ts*1e3:.2f} ms")
                col2.metric("Max Shift (Δt_max)", f"{dt_max*1e3:.2f} ms")
                col3.metric("Sensitivity (k_ppm)", f"{k_ppm:.2f}")

            elif metric_active == "Guard Interval & Collision":
                st.markdown("### Guard Interval & Collision Prevention")
                st.markdown(r"""
1. To avoid pulse overlap between adjacent clock periods, the maximum position shift plus pulse width must be strictly less than $T_s$:

$$
T_{guard} = T_s - (\tau_0 + \Delta t_{max} + \tau) > 0
$$
""")
                max_pos = nominal_delay + k_ppm * Ts + tau_ppm
                guard_time = Ts - max_pos
                col1, col2, col3 = st.columns(3)
                col1.metric("Max End Position", f"{max_pos*1e3:.2f} ms")
                col2.metric("Guard Time (T_guard)", f"{guard_time*1e3:.2f} ms")
                col3.metric("Collision Status", f"{'❌ OVERLAP' if guard_time <= 0 else '✅ SAFE'}")

            elif metric_active == "Constant Power Property":
                st.markdown("### Constant Average Transmitted Power")
                st.markdown(r"""
1. Unlike PAM and PWM, **PPM transmitted power remains strictly constant** regardless of message signal $m(t)$!
2. Fixed pulse duration $\tau$ and fixed pulse amplitude $A_c$:

$$
P_{avg} = \frac{\tau}{T_s} A_c^2 = D_{PPM} \cdot A_c^2 \quad \text{(Constant)}
$$
""")
                p_avg_ppm = (tau_ppm / Ts) * (Ac**2)
                col1, col2, col3 = st.columns(3)
                col1.metric("Fixed Width (τ)", f"{tau_ppm*1e3:.2f} ms")
                col2.metric("Duty Cycle (D_PPM)", f"{(tau_ppm/Ts)*100:.1f} %")
                col3.metric("Const Average Power", f"{p_avg_ppm:.3f} W")

            elif metric_active == "Noise Immunity & Jitter":
                st.markdown("### Noise Immunity & Pulse Jitter Sensitivity")
                st.markdown(r"""
1. Because signal information is encoded solely in pulse **timing/position**, amplitude noise can be clipped using a amplitude limiter receiver.
2. Noise only causes timing jitter $\delta t$:

$$
\sigma_t = \frac{\sigma_n}{\left| \frac{ds(t)}{dt} \right|}
$$
""")
                bw_ppm = 1.0 / (2 * tau_ppm)
                col1, col2 = st.columns(2)
                col1.metric("Noise Immunity", "Superior to PAM & PWM")
                col2.metric("Required Channel BW", f"{bw_ppm:.1f} Hz")

    # -----------------------------
    # Theory & Derivations - PPM
    # -----------------------------
    st.markdown("## About Pulse Position Modulation (PPM)")
    st.markdown(r"""
- **Pulse Position Modulation (PPM)** varies the **position (time delay)** of fixed-duration pulses relative to reference clock positions $k T_s$, proportional to the modulating signal $m(t)$.
- **Generation:** PPM is typically generated from PWM! The falling edge (trailing edge) of a PWM pulse triggers a monostable multivibrator (one-shot) that generates a pulse of fixed width $\tau$.
- **Demodulation:** Standard receiver uses an **SR Flip-Flop** (Set by reference clock pulse at $k T_s$, Reset by incoming PPM pulse arrival) to convert PPM back into a PWM wave, which is then passed through an LPF.
- **Key Advantage:** Transmitted power is constant ($P_{avg} = \frac{\tau}{T_s} A_c^2$) regardless of message amplitude, allowing transmitter power amplifiers to run at peak efficiency without distortion.
- **Applications:** Optical fiber communications, RF remote controls, impulse radio ultra-wideband (IR-UWB), and deep-space telemetry.
""")

    st.subheader("PPM Derivations & Mathematical Foundations")
    derivation_active = derivation_buttons("ppm_active_derivation", ["PPM Pulse Position Expression", "Generation from PWM Trailing Edge", "SR Flip-Flop Demodulation", "Constant Power & SNR Analysis"])

    if derivation_active:
        with st.container(border=True):
            if derivation_active == "PPM Pulse Position Expression":
                st.markdown("### Mathematical Expression for PPM Waveform")
                st.markdown(r"""
**1. Pulse Delay:**
The starting instant $t_k$ of the $k$-th PPM pulse is:

$$
t_k = k T_s + \tau_0 + k_{PPM} m(k T_s)
$$

where:
- $k T_s$ is the $k$-th reference clock tick.
- $\tau_0$ is the unmodulated nominal offset delay.
- $k_{PPM}$ is the position modulation sensitivity.

**2. Time-Domain Waveform $s_{PPM}(t)$:**
For pulses of fixed duration $\tau$ and amplitude $A_c$:

$$
s_{PPM}(t) = \sum_{k=-\infty}^{\infty} A_c \cdot \text{rect}\left( \frac{t - t_k - \tau/2}{\tau} \right)
$$
""")

            elif derivation_active == "Generation from PWM Trailing Edge":
                st.markdown("### Generation of PPM from PWM Trailing Edge")
                st.markdown(r"""
**1. Block Diagram Mechanics:**
1. Message signal $m(t)$ and carrier ramp are fed into a comparator to generate trailing-edge PWM.
2. The trailing-edge transition of the PWM wave occurs at time $t_k = k T_s + \tau_k$, where $\tau_k = \tau_0 + k_{PWM} m(k T_s)$.
3. This trailing edge triggers a negative-edge monostable multivibrator (one-shot).

**2. Monostable Output:**
The monostable output remains HIGH for a fixed time interval $\tau$, producing the exact PPM pulse:

$$
s_{PPM}(t) = \text{Monostable}\left( \frac{d}{dt} s_{PWM}(t) \right)
$$
""")

            elif derivation_active == "SR Flip-Flop Demodulation":
                st.markdown("### SR Flip-Flop Demodulation Derivation")
                st.markdown(r"""
**1. Receiver Flip-Flop Operations:**
- **Set (S) input:** Synchronized periodic clock pulse train at $t = k T_s$. Sets $Q = \text{HIGH} \, (A_c)$.
- **Reset (R) input:** Received PPM pulse at $t = t_k$. Resets $Q = \text{LOW} \, (0)$.

**2. Regenerated Output:**
The output $Q(t)$ of the flip-flop stays HIGH from $t = k T_s$ to $t = t_k$.
The pulse width of $Q(t)$ is:

$$
W_k = t_k - k T_s = \tau_0 + k_{PPM} m(k T_s)
$$

This is an exact PWM wave!

**3. Low-Pass Filter Output:**
Passing $Q(t)$ through an LPF with cutoff $f_m < f_{cut} < f_s - f_m$ extracts the average voltage:

$$
v_{LPF}(t) = A_c \frac{\tau_0}{T_s} + \frac{A_c k_{PPM}}{T_s} m(t)
$$

Removing the DC term recovers the original message $m(t)$ perfectly!
""")

            elif derivation_active == "Constant Power & SNR Analysis":
                st.markdown("### Constant Power & Noise Performance Derivation")
                st.markdown(r"""
**1. Constant Power Proof:**
Since every PPM pulse has identical amplitude $A_c$ and fixed width $\tau$, the energy per pulse is $E_p = A_c^2 \tau$.
Average power over any period $T_s$:

$$
P_{avg} = \frac{E_p}{T_s} = \frac{\tau}{T_s} A_c^2
$$

Because $P_{avg}$ is completely independent of message $m(t)$, the transmitter operates at constant power.

**2. Figure of Merit (SNR Advantage):**
In PPM, noise introduces timing error (jitter) $\delta t$. The signal-to-noise ratio at output is:

$$
\left(\frac{S}{N}\right)_{out} = \frac{1}{2} \left( \frac{\Delta t_{max}}{\tau_{edge}} \right)^2 \left(\frac{S}{N}\right)_{channel}
$$

where $\tau_{edge}$ is the pulse rise time. Since $\tau_{edge} \ll \Delta t_{max}$, PPM provides significant SNR improvement over PAM!
""")


# ==============================================================================
# COMPREHENSIVE COMPARATIVE TABLE
# ==============================================================================
st.markdown("---")
st.markdown("## Comparative Analysis: PAM vs PWM vs PPM")
st.markdown(
    r"""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}
th, td {
    border: 1px solid #444;
    padding: 10px 14px;
    text-align: left;
}
th {
    background-color: #2b2b2b;
    color: #ffffff;
}
</style>

<table>
    <thead>
        <tr>
            <th>Parameter / Feature</th>
            <th>Pulse Amplitude Modulation (PAM)</th>
            <th>Pulse Width Modulation (PWM)</th>
            <th>Pulse Position Modulation (PPM)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>Modulated Parameter</b></td>
            <td>Pulse Amplitude (Height)</td>
            <td>Pulse Width (Duration / Length)</td>
            <td>Pulse Position (Time Shift / Delay)</td>
        </tr>
        <tr>
            <td><b>Pulse Width</b></td>
            <td>Fixed ($\tau$)</td>
            <td>Variable ($\tau(t) = \tau_0 + k \cdot m(t)$)</td>
            <td>Fixed ($\tau$)</td>
        </tr>
        <tr>
            <td><b>Pulse Amplitude</b></td>
            <td>Variable ($A(t) \propto m(t)$)</td>
            <td>Fixed ($A_c$)</td>
            <td>Fixed ($A_c$)</td>
        </tr>
        <tr>
            <td><b>Transmitted Power</b></td>
            <td>Varies with message signal amplitude</td>
            <td>Varies with pulse width (duty cycle)</td>
            <td><b>Constant</b> ($P_{avg} = \frac{\tau}{T_s} A_c^2$)</td>
        </tr>
        <tr>
            <td><b>Noise Immunity</b></td>
            <td>Poor (susceptible to amplitude noise)</td>
            <td>Moderate (amplitude limiters remove noise)</td>
            <td><b>Superior</b> (only timing jitter affects output)</td>
        </tr>
        <tr>
            <td><b>Bandwidth Requirement</b></td>
            <td>Lowest among pulse modes ($BW \approx 1/2\tau$)</td>
            <td>Moderate ($BW \approx 1/2\tau_{min}$)</td>
            <td>Highest ($BW \approx 1/2\tau_{ppm}$)</td>
        </tr>
        <tr>
            <td><b>Receiver Synchronization</b></td>
            <td>Not mandatory for basic envelope recovery</td>
            <td>Not mandatory (direct LPF demodulation)</td>
            <td><b>Mandatory</b> (requires synchronized clock reference)</td>
        </tr>
        <tr>
            <td><b>Circuit Complexity</b></td>
            <td>Simplest (Sample & Hold circuit)</td>
            <td>Moderate (Comparator + Sawtooth Ramp)</td>
            <td>Higher (PWM + Monostable + SR Flip-Flop)</td>
        </tr>
        <tr>
            <td><b>Primary Applications</b></td>
            <td>Pre-stage to PCM (digitization), FDM telemetry</td>
            <td>Motor speed control, Class-D audio, DC-DC power converters</td>
            <td>Optical communications, RF remote control, UWB radar</td>
        </tr>
    </tbody>
</table>
""",
    unsafe_allow_html=True,
)
