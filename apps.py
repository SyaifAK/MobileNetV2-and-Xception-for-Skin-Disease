import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from streamlit_option_menu import option_menu
from streamlit_cropper import st_cropper


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, div, span, h1, h2, h3, h4, h5, h6, p, a, li, ul, button, input, label {
        font-family: 'Poppins' !important;
    }
  
    
    </style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_models():
    model_mobilenet = load_model("mobilenetv2last.keras")
    model_xception = load_model("xceptionlast.keras")
    return model_mobilenet, model_xception 
model_mobilenet, model_xception = load_models()

# Daftar label
class_labels = ['eksim', 'herpes', 'melanoma', 'normal']

# Fungsi deskripsi penyakit
def disease_description(label):
    deskripsi = {
        "eksim": (
            "Eksim (dermatitis atopik) adalah peradangan pada kulit yang menyebabkan rasa gatal, merah, kering, bahkan pecah-pecah. Sering terjadi pada anak-anak dan bisa kambuh.\n\n"
            "Eksim dapat diatasi dengan menjaga kelembapan kulit, menghindari pemicu iritasi, serta menggunakan obat yang sesuai. Gunakan pelembap secara rutin setelah mandi, hindari sabun yang mengandung bahan kimia keras, dan kenakan pakaian berbahan lembut seperti katun. Jika gatal parah, bisa digunakan salep kortikosteroid atau obat antihistamin sesuai anjuran dokter. Kurangi stres, cukup istirahat, dan hindari makanan pemicu alergi jika ada. Segera konsultasi ke dokter jika eksim memburuk atau disertai infeksi.\n\n"
            "Pertimbangkan untuk berkonsultasi langsung dengan tenaga medis untuk diagnosis yang lebih akurat."
        ),
        "herpes": (
            "Herpes kulit disebabkan oleh virus Herpes Simplex. Umumnya muncul sebagai lepuhan kecil yang terasa gatal atau nyeri, biasanya di sekitar mulut atau area genital.\n\n"
            "Untuk mengatasinya, penting menjaga kebersihan area yang terinfeksi dan menghindari menyentuh atau menggaruk lepuhan. Obat antivirus seperti asiklovir (acyclovir) dapat digunakan dalam bentuk salep atau tablet sesuai resep dokter untuk mempercepat penyembuhan dan mencegah penyebaran. Istirahat cukup, konsumsi makanan bergizi, dan hindari stres berlebihan agar daya tahan tubuh tetap kuat. Hindari kontak langsung dengan kulit orang lain selama luka aktif agar tidak menular.\n\n"        
            "Pertimbangkan untuk berkonsultasi langsung dengan tenaga medis untuk diagnosis yang lebih akurat."
        ),
        "melanoma": (
            "Melanoma adalah jenis kanker kulit yang berbahaya dan berkembang dari sel penghasil pigmen (melanosit). Deteksi dini sangat penting karena melanoma bisa menyebar dengan cepat.\n\n"
            "Penanganannya harus segera dilakukan oleh dokter spesialis kulit atau onkologi. Pengobatan utama biasanya melalui operasi pengangkatan jaringan kanker, dan bisa dilanjutkan dengan terapi tambahan seperti imunoterapi, kemoterapi, atau terapi target jika sudah menyebar. Deteksi dini sangat penting—melanoma sering muncul sebagai tahi lalat yang berubah bentuk, warna, , rasa gatal atau ukuran yang membesar dengan cepat. Hindari paparan sinar UV berlebihan, gunakan tabir surya, dan periksa kulit secara berkala untuk mendeteksi tanda awal melanoma.\n\n"
            "Pertimbangkan untuk berkonsultasi langsung dengan tenaga medis untuk diagnosis yang lebih akurat."
        ),
        "normal": "Kulit dalam kondisi normal tanpa tanda-tanda penyakit kulit yang mencolok. Namun, tetap jaga kebersihan dan kelembapan kulit."
        
    }
    return deskripsi.get(label, "Informasi belum tersedia untuk label ini.")

# Fungsi prediksi
def predict_image(image_pil):
    # Konversi ke array numpy
    img_array = np.array(image_pil.convert('RGB').resize((224, 224)))
    # Tambah dimensi batch
    input_array = np.expand_dims(img_array, axis=0)
 

    #mobilenet prediction
    input_mobilenet = mobilenet_preprocess(input_array.copy())
    prediction_mobilenet = model_mobilenet.predict(input_mobilenet)
    label_mobilenet = class_labels[np.argmax(prediction_mobilenet)]
    confidence_mobilenet = np.max(prediction_mobilenet)
    # GradCAM 
    heatmap_mobilenet = get_gradcam_heatmap(input_mobilenet, model_mobilenet, 'Conv_1')
    overlay_mobilenet = overlay_heatmap(heatmap_mobilenet, img_array.copy())

    
    #xception prediction
    input_xception = xception_preprocess(input_array.copy())
    prediction_xception = model_xception.predict(input_xception)
    label_xception = class_labels[np.argmax(prediction_xception)]
    confidence_xception = np.max(prediction_xception)
    #gradcam
    heatmap_xception = get_gradcam_heatmap(input_xception, model_xception, 'block14_sepconv2_act')
    overlay_xception = overlay_heatmap(heatmap_xception, img_array.copy())

    return {
        'mobilenet': (label_mobilenet, confidence_mobilenet, heatmap_mobilenet, overlay_mobilenet),
        'xception': (label_xception, confidence_xception, heatmap_xception, overlay_xception)
    }
    


#fungsi grad cam
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = 1.0 - heatmap
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    output = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return output


with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",  # Tidak perlu judul kalau mau tampil rapi
        options=["Beranda", "Deteksi Penyakit Kulit", "Model", "Profil"],
        icons=["house", "camera", "cpu", "person"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "10!important", "background-color": ""},
            "icon": {"color": "white", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "rgba(255, 255, 255, 0.1)",
                "color": "white",
                "font-family": 'Poppins' 
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 255, 255, 0.2)",
                "color": "white",
                "font-weight": "normal"
            },
        }
     
    )


# Halaman Beranda
if selected == "Beranda":
    # Judul Utama
    st.markdown("""
    <h1 style='text-align: center;'>
        🩺 Deteksi Penyakit Kulit dengan Deep Learning
    </h1>
    """, unsafe_allow_html=True)

    # Gambar ilustrasi
    st.image("1.png", caption="🔍 Ilustrasi penyakit kulit", use_container_width=True)

    st.markdown("---")

    # Dua kolom dalam gaya seperti tabel dengan dark background
    st.markdown("""
    <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 280px; background-color: #2c2c2c; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.4);'>
            <h4 style='color: white;'>🤖 Model Deep Learning</h4>
            <ul style='color: white; margin: 0; padding-left: 20px;'>
                <li>MobileNetV2</li>
                <li>Xception</li>
            </ul>
        </div>
        <div style='flex: 1; min-width: 280px; background-color: #2c2c2c; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.4);'>
            <h4 style='color: white;'>🎯 Tujuan Aplikasi</h4>
            <p style='color: white; margin: 0;'>Membantu proses awal identifikasi penyakit kulit:</p>
            <ul style='color: white; margin: 0; padding-left: 20px;'>
                <li>Eksim</li>
                <li>Herpes</li>
                <li>Melanoma</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Disclaimer
    with st.container():
        st.subheader("⚠️ Disclaimer")
        st.markdown("""
        <div style='
            background-color: #DC2525;
            padding: 15px 20px;
            border-left: 5px solid #ffa500;
            border-radius: 6px;
            font-size: 16px;
            line-height: 1.6;
            color: white;
        '>
            Hasil dari aplikasi ini bukan merupakan <strong>diagnosis medis resmi</strong>.  
            Untuk kepastian lebih lanjut, silakan konsultasikan dengan dokter spesialis kulit.
        </div>
        """, unsafe_allow_html=True)




# Halaman Deteksi
elif selected == "Deteksi Penyakit Kulit":
    st.title("📷 Deteksi Penyakit Kulit")
    st.markdown("### 📝 Petunjuk Penggunaan")
    st.info("""
    - Pastikan gambar memiliki **pencahayaan yang bagus, kualitas yang baik** dan fokus pada area kulit yang ingin dideteksi.
    - **Ukuran minimal gambar:** 224 x 224 piksel
    - Format file: JPG, JPEG, atau PNG
    """)

    st.markdown("----")
    st.markdown("### 📤 Unggah Gambar")

    uploaded_file = st.file_uploader("Pilih gambar kulit untuk dideteksi", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        width, height = image.size

        if width <= 224 or height <= 224:
            st.error(f"❌ Ukuran gambar terlalu kecil ({width}x{height}). "
                     "Gunakan gambar minimal 224x224 piksel.")
        else:
            st.markdown("### ✂️ Crop Gambar")

            cropped_image = st_cropper(
                image,
                realtime_update=True,
                box_color='#FF4B4B',
                aspect_ratio=(1, 1),
            )

            if cropped_image.width < 120 or cropped_image.height < 120:
                st.warning("⚠️ Area crop terlalu kecil. Silakan crop ulang dengan area yang lebih besar.")
            else:
                st.image(cropped_image, caption="🖼️ Gambar Setelah Crop", use_container_width=True)

                if st.button("🚀 Lakukan Prediksi"):
                    with st.spinner("🔍 Mendeteksi..."):
                        resized_image = cropped_image.resize((224, 224), Image.LANCZOS).convert('RGB')
                        img_array = np.array(resized_image).astype('float32') / 255.0
                        input_array = np.expand_dims(img_array, axis=0)

                        # Prediksi
                        result = predict_image(resized_image)

                    # ===== MobileNetV2 Result =====
                    st.markdown("## 🤖 Hasil Prediksi MobileNetV2")
                    label_mobilenet, confidence_mobilenetv2, heatmap_mobilenet, overlay_mobilenet = result["mobilenet"]

                    if confidence_mobilenetv2 < 0.7:
                        st.warning("⚠️ Tingkat keyakinan model MobileNetV2 di bawah 70%. Gambar mungkin tidak sesuai dengan petunjuk penggunaan.")
                    else:
                        st.success(f"✅ Teridentifikasi: **{label_mobilenet.capitalize()}** ({confidence_mobilenetv2*100:.2f}%)")
                        st.info(disease_description(label_mobilenet))

                        st.markdown("### 🔥 Grad-CAM MobileNetV2")
                        heatmap_mobilenet = get_gradcam_heatmap(input_array, model_mobilenet, "Conv_1")
                        gradcam_img_mobilenet = overlay_heatmap(heatmap_mobilenet, np.array(resized_image))
                        st.image(gradcam_img_mobilenet, caption="Grad-CAM MobileNetV2", use_container_width=True)

                    st.markdown("---")

                    # ===== Xception Result =====
                    st.markdown("## 🤖 Hasil Prediksi Xception")
                    label_xception, confidence_xception, heatmap_xception, overlay_xception = result["xception"]

                    if confidence_xception < 0.7:
                        st.warning("⚠️ Tingkat keyakinan model Xception di bawah 70%. Gambar mungkin tidak sesuai dengan petunjuk penggunaan.")
                    else:
                        st.success(f"✅ Teridentifikasi: **{label_xception.capitalize()}** ({confidence_xception*100:.2f}%)")
                        st.info(disease_description(label_xception))

                        st.markdown("### 🔥 Grad-CAM Xception")
                        heatmap_xception = get_gradcam_heatmap(input_array, model_xception, "block14_sepconv2_act")
                        gradcam_img_xception = overlay_heatmap(heatmap_xception, np.array(resized_image))
                        st.image(gradcam_img_xception, caption="Grad-CAM Xception", use_container_width=True)

                else:
                    st.warning("Klik tombol **Lakukan Prediksi** untuk memulai proses deteksi.")




elif selected == "Model":
    st.markdown("<h1 style='text-align: center;'>🔬 Tentang Model</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        """
        <div style='display: flex; gap: 20px;'>
            <div style='flex: 1; background-color: #2c2c2c; padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.4); color: white; display: flex; flex-direction: column;'>
                <h3 style='margin-top: 0;'>📱 MobileNetV2</h3>
                <ul style='padding-left: 20px; line-height: 1.8; flex-grow: 1;'>
                    <li><strong>⚡ Cepat dan ringan</strong>: Cocok untuk perangkat biasa seperti HP/laptop tanpa hardware mahal.</li>
                    <li><strong>🔋 Hemat daya dan memori</strong>: Efisien, tidak banyak makan RAM/baterai.</li>
                    <li><strong>🌐 Ideal untuk aplikasi mobile/web</strong>: Respon cepat & performa optimal.</li>
                </ul>
            </div>
            <div style='flex: 1; background-color: #2c2c2c; padding: 25px; border-radius: 12px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.4); color: white; display: flex; flex-direction: column;'>
                <h3 style='margin-top: 0;'>🧠 Xception</h3>
                <ul style='padding-left: 20px; line-height: 1.8; flex-grow: 1;'>
                    <li><strong>🔍 Bagus untuk gambar detail</strong>: Kuat mengenali gambar yang kompleks.</li>
                    <li><strong>⚙️ Struktur efisien & cerdas</strong>: Model dalam tapi tetap ringan secara arsitektur.</li>
                    <li><strong>🎯 Hasil lebih presisi</strong>: Cocok untuk klasifikasi gambar yang menuntut akurasi tinggi.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.4); color: white; font-size: 16px;'>
            Kedua model telah dioptimalkan menggunakan <strong>Transfer Learning</strong> dan <strong>Fine Tuning</strong> 
            untuk mendapatkan <strong>hyperparameter terbaik</strong> demi meningkatkan akurasi dan efisiensi.
        </div>
        """, unsafe_allow_html=True
    )


elif selected == "Profil":
    st.markdown("<h1 style='text-align: center;'>👩‍💻 Profil Peneliti</h1>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
    <style>
    .profil-box {
        background-color: #2c2c2c;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        color: white;
        font-size: 17px;
        line-height: 1.8;
        margin-top: 20px;
        width: 100%;
        box-sizing: border-box;
    }
    .profil-item {
        margin-bottom: 10px;
    }
    .profil-highlight {
        font-weight: bold;
        color: #ffffff;
    }
    </style>
    <div class="profil-box">
        <div class="profil-item"><span class="profil-highlight">🧑 Nama:</span> Syaif Al Khalim</div>
        <div class="profil-item"><span class="profil-highlight">🆔 NIM:</span> 21670040</div>
        <div class="profil-item"><span class="profil-highlight">🎓 Program Studi:</span> Informatika</div>
        <div class="profil-item"><span class="profil-highlight">🏫 Universitas:</span> Universitas PGRI Semarang</div>
        <hr style='border-top: 1px solid #555; margin: 20px 0;'>
        <div class="profil-item">Aplikasi ini dikembangkan sebagai bagian dari <strong>penelitian skripsi</strong> dalam bidang <strong>klasifikasi citra penyakit kulit</strong> berbasis <strong>deep learning</strong>.</div>
    </div>
    """, unsafe_allow_html=True)

