import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import librosa
import joblib
import os
import python_speech_features as psf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame

# ==========================================
# CONFIGURATION
# ==========================================
# Map your Model Names -> Google Speech Recognition Codes
LANG_MAPPING = {
    "English": "en-US",
    "French": "fr-FR",
    "Darija": "ar-AR",
    "Nederlands": "nl-NL",
    "Japanese": "ja-JP",
}


# ==========================================
# 1. BACKEND LOGIC
# ==========================================
class AudioAnalyzer:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Loads all available GMM models from the trained_models directory
        that match the language names in LANG_MAPPING.
        Checks for both .joblib and .gmm extensions.
        """
        print("Loading Language Models...")
        loaded_count = 0

        for lang in LANG_MAPPING.keys():
            model_loaded = False

            # List of possible filenames to try
            possible_filenames = [
                f"trained_models/GMM_{lang}_best.joblib",
                f"trained_models/GMM_{lang}_best.gmm",
            ]

            for filename in possible_filenames:
                if os.path.exists(filename):
                    try:
                        self.models[lang] = joblib.load(filename)
                        print(f" -> Loaded: {lang} (from {os.path.basename(filename)})")
                        loaded_count += 1
                        model_loaded = True
                        break  # Stop checking other extensions for this language
                    except Exception as e:
                        print(f" -> Error loading {filename}: {e}")

            if not model_loaded:
                print(
                    f" -> Warning: No model file found for {lang} (checked .joblib and .gmm)"
                )

        if loaded_count == 0:
            messagebox.showwarning(
                "Warning",
                "No language models were loaded!\nMake sure files are named 'GMM_English_best.joblib', etc.",
            )

    def preprocess_audio(self, file_path):
        """
        Extracts MFCCs + Deltas and removes silence (Hybrid Logic)
        """
        # Load with original sampling rate
        signal, sr = librosa.load(file_path, sr=None)

        # 1. Extract MFCC
        n_fft = 1024 if sr <= 16000 else 2048
        mfccs = psf.mfcc(
            signal, sr, numcep=13, nfft=n_fft, winfunc=np.hamming, appendEnergy=False
        )

        # 2. Silence Removal (Hybrid GMM + KMeans)
        energies = np.sum(np.square(mfccs), axis=1).reshape(-1, 1)

        # KMeans
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
        kmeans.fit(energies)
        kmeans_centers = kmeans.cluster_centers_.ravel()

        # GMM
        gmm = GaussianMixture(n_components=2, random_state=42, init_params="k-means++")
        gmm.fit(energies)
        gmm_means = gmm.means_.ravel()

        # Hybrid Threshold
        threshold = (np.min(gmm_means) + np.min(kmeans_centers)) / 2

        is_speech = (energies > threshold).flatten()
        clean_mfcc = mfccs[is_speech]

        # Safety check for very short audio
        if len(clean_mfcc) < 2:
            return None

        # 3. Deltas
        deltas = psf.delta(clean_mfcc, 2)
        features = np.hstack((clean_mfcc, deltas))

        return features

    def identify_language(self, file_path):
        """
        Compares audio against loaded Language Models
        """
        if not self.models:
            return "Unknown", "No models loaded."

        features = self.preprocess_audio(file_path)
        if features is None:
            return "Error", "Audio too short or silent."

        best_score = -float("inf")
        winner = "Unknown"
        scores_text = ""

        # Score against every loaded language model
        for label, model in self.models.items():
            try:
                score = model.score(features)
                scores_text += f"{label}: {score:.2f}\n"

                if score > best_score:
                    best_score = score
                    winner = label
            except:
                scores_text += f"{label}: Error\n"

        return winner, scores_text

    def speech_to_text(self, file_path, lang_code):
        r = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            # Clean noise slightly
            # r.adjust_for_ambient_noise(source)
            audio_data = r.record(source)
            try:
                # Use the detected language code (e.g., 'fr-FR')
                text = r.recognize_google(audio_data, language=lang_code)
                return text
            except sr.UnknownValueError:
                return "(Audio unclear or empty)"
            except sr.RequestError:
                return "(API Connection Error)"

    def translate_text(self, text, source_lang_name):
        """
        Smart translation:
        - If input is English -> Translate to French
        - If input is French/Other -> Translate to English
        """
        try:
            if source_lang_name == "English":
                target = "fr"  # To French
            else:
                target = "en"  # To English

            translated = GoogleTranslator(source="auto", target=target).translate(text)
            return translated, target
        except Exception as e:
            return f"Translation Error: {e}", "en"

    def text_to_speech(self, text, lang_code):
        try:
            # gTTS requires simplified codes (en, fr, ar), not (en-US)
            # We strip the region code if present
            simple_lang = lang_code.split("-")[0] if "-" in lang_code else lang_code

            tts = gTTS(text=text, lang=simple_lang, slow=False)
            save_path = "translation_output.mp3"
            tts.save(save_path)
            return save_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None


# ==========================================
# 2. FRONTEND GUI
# ==========================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Identification & Translation System")
        self.root.geometry("650x750")

        self.analyzer = AudioAnalyzer()
        self.current_file = None
        self.detected_language = None  # Stores the result of Step 2

        # Header
        tk.Label(
            root,
            text="Multilingual Audio System",
            font=("Arial", 18, "bold"),
            fg="#333",
        ).pack(pady=15)

        # 1. Load Button
        frame1 = tk.Frame(root, pady=5)
        frame1.pack(fill="x", padx=20)
        tk.Button(
            frame1,
            text="1. Load Audio File",
            command=self.load_file,
            bg="#ADD8E6",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(fill="x")
        self.lbl_filename = tk.Label(frame1, text="No file loaded", fg="gray")
        self.lbl_filename.pack()

        # 2. Identify Button
        frame2 = tk.Frame(root, pady=5)
        frame2.pack(fill="x", padx=20)
        tk.Button(
            frame2,
            text="2. Identify Language",
            command=self.run_id,
            bg="#90EE90",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(fill="x")
        self.lbl_result = tk.Label(
            frame2, text="Detected: ...", font=("Arial", 14, "bold"), fg="blue"
        )
        self.lbl_result.pack(pady=5)

        # 3. Process Button
        frame3 = tk.Frame(root, pady=5)
        frame3.pack(fill="x", padx=20)
        tk.Button(
            frame3,
            text="3. ASR (Speech-to-Text) & Translate",
            command=self.run_asr_trans,
            bg="#FFD700",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(fill="x")

        # Text Areas
        tk.Label(
            root,
            text="Transcribed Text (Original Language):",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=20, pady=(10, 0))
        self.txt_original = scrolledtext.ScrolledText(
            root, height=5, font=("Arial", 10)
        )
        self.txt_original.pack(fill="x", padx=20)

        tk.Label(root, text="Translated Text:", font=("Arial", 10, "bold")).pack(
            anchor="w", padx=20, pady=(10, 0)
        )
        self.txt_translated = scrolledtext.ScrolledText(
            root, height=5, font=("Arial", 10)
        )
        self.txt_translated.pack(fill="x", padx=20)

        # 4. Play Button
        frame4 = tk.Frame(root, pady=20)
        frame4.pack(fill="x", padx=20)
        tk.Button(
            frame4,
            text="4. Speak Translation",
            command=self.play_translation,
            bg="#FFB6C1",
            font=("Arial", 10, "bold"),
            height=2,
        ).pack(fill="x")

        # Init Audio
        pygame.mixer.init()

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.flac *.mp3")]
        )
        if file_path:
            self.current_file = file_path
            self.lbl_filename.config(text=os.path.basename(file_path))
            # Reset UI
            self.detected_language = None
            self.lbl_result.config(text="Detected: ...")
            self.txt_original.delete(1.0, tk.END)
            self.txt_translated.delete(1.0, tk.END)

    def run_id(self):
        if not self.current_file:
            messagebox.showwarning("Warning", "Please load an audio file first.")
            return

        winner, details = self.analyzer.identify_language(self.current_file)

        self.detected_language = winner
        self.lbl_result.config(text=f"Detected: {winner}")
        messagebox.showinfo("Log-Likelihood Scores", details)

    def run_asr_trans(self):
        if not self.current_file:
            return
        if not self.detected_language or self.detected_language == "Unknown":
            messagebox.showwarning(
                "Warning", "Please run Language Identification (Step 2) first."
            )
            return

        # Get the correct language code (e.g., 'fr-FR') for the detected language
        lang_code = LANG_MAPPING.get(
            self.detected_language, "en-US"
        )  # Default to English if unknown

        self.root.config(cursor="wait")
        self.root.update()

        # 1. Speech to Text (using detected lang)
        text = self.analyzer.speech_to_text(self.current_file, lang_code)
        self.txt_original.delete(1.0, tk.END)
        self.txt_original.insert(tk.END, text)

        # 2. Translation
        # If detected English -> Translate to French. Else -> English.
        trans_text, self.target_lang_code = self.analyzer.translate_text(
            text, self.detected_language
        )

        self.txt_translated.delete(1.0, tk.END)
        self.txt_translated.insert(tk.END, trans_text)

        self.root.config(cursor="")

    def play_translation(self):
        text = self.txt_translated.get("1.0", tk.END).strip()
        if not text:
            return

        # We use the target language code calculated in the translation step
        target = getattr(self, "target_lang_code", "en")

        file_path = self.analyzer.text_to_speech(text, target)
        if file_path:
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
            except Exception as e:
                messagebox.showerror("Audio Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
