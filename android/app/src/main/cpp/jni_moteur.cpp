// Pont JNI entre l'application (MoteurNatif.kt) et le moteur C++ (moteur.h).
// Les textes passent en octets UTF-8 (ByteArray) : pas de conversion « UTF-8 modifié » de JNI,
// qui abîmerait les émojis et certains caractères.
#include <jni.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "moteur.h"

namespace {
std::mutex g_erreur_verrou;
std::string g_erreur;

void definir_erreur(const std::string & e) {
    std::lock_guard<std::mutex> l(g_erreur_verrou);
    g_erreur = e;
}

std::string octets(JNIEnv * env, jbyteArray tableau) {
    if (!tableau) return "";
    const jsize n = env->GetArrayLength(tableau);
    std::string s(static_cast<size_t>(n), '\0');
    env->GetByteArrayRegion(tableau, 0, n, reinterpret_cast<jbyte *>(s.data()));
    return s;
}

jbyteArray vers_octets(JNIEnv * env, const std::string & s) {
    jbyteArray t = env->NewByteArray(static_cast<jsize>(s.size()));
    env->SetByteArrayRegion(t, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte *>(s.data()));
    return t;
}

std::string chaine(JNIEnv * env, jstring j) {
    const char * c = env->GetStringUTFChars(j, nullptr);
    std::string s(c ? c : "");
    if (c) env->ReleaseStringUTFChars(j, c);
    return s;
}

mirador::Moteur * moteur(jlong h) { return reinterpret_cast<mirador::Moteur *>(h); }
}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_miradorsi_app_MoteurNatif_initialiser(JNIEnv * env, jobject, jstring dossier) {
    // variantes du moteur de calcul (selon les instructions du processeur) livrées avec l'appli
    ggml_backend_load_all_from_path(chaine(env, dossier).c_str());
}

JNIEXPORT jlong JNICALL Java_com_miradorsi_app_MoteurNatif_ouvrir(JNIEnv * env, jobject, jstring chemin, jint n_ctx,
                                                                 jint n_threads) {
    std::string erreur;
    auto m = mirador::Moteur::ouvrir(chaine(env, chemin), n_ctx, n_threads, erreur);
    if (!m) {
        definir_erreur(erreur);
        return 0;
    }
    return reinterpret_cast<jlong>(m.release());
}

JNIEXPORT jstring JNICALL Java_com_miradorsi_app_MoteurNatif_derniereErreur(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> l(g_erreur_verrou);
    return env->NewStringUTF(g_erreur.c_str());
}

JNIEXPORT jstring JNICALL Java_com_miradorsi_app_MoteurNatif_description(JNIEnv * env, jobject, jlong h) {
    const std::string d = moteur(h)->architecture() + " · " + std::to_string(moteur(h)->parametres());
    return env->NewStringUTF(d.c_str());
}

JNIEXPORT jbyteArray JNICALL Java_com_miradorsi_app_MoteurNatif_formater(JNIEnv * env, jobject, jlong h,
                                                                        jobjectArray roles, jobjectArray contenus) {
    std::vector<mirador::Message> messages;
    const jsize n = env->GetArrayLength(roles);
    for (jsize i = 0; i < n; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto contenu = static_cast<jbyteArray>(env->GetObjectArrayElement(contenus, i));
        messages.push_back({chaine(env, role), octets(env, contenu)});
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(contenu);
    }
    return vers_octets(env, moteur(h)->formater(messages));
}

JNIEXPORT jbyteArray JNICALL Java_com_miradorsi_app_MoteurNatif_generer(JNIEnv * env, jobject, jlong h,
                                                                       jbyteArray prompt, jint max_tokens,
                                                                       jfloat temperature, jobject rappel) {
    jclass classe = env->GetObjectClass(rappel);
    jmethodID methode = env->GetMethodID(classe, "morceau", "([B)Z");
    std::string erreur;
    const std::string texte = moteur(h)->generer(
        octets(env, prompt), max_tokens, temperature,
        [&](const std::string & morceau) {
            jbyteArray t = vers_octets(env, morceau);
            const jboolean continuer = env->CallBooleanMethod(rappel, methode, t);
            env->DeleteLocalRef(t);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                return false;
            }
            return continuer == JNI_TRUE;
        },
        erreur);
    if (!erreur.empty()) definir_erreur(erreur);
    return vers_octets(env, texte);
}

JNIEXPORT void JNICALL Java_com_miradorsi_app_MoteurNatif_fermer(JNIEnv *, jobject, jlong h) {
    delete moteur(h);
}

}  // extern "C"
