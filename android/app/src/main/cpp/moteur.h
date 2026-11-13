// Moteur de génération de Mirador SI pour les modèles ouverts (format GGUF), sur llama.cpp.
// Le même code sert à l'application Android (via jni_moteur.cpp) et aux essais sur PC.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct common_chat_templates;

namespace mirador {

struct Message {
    std::string role;     // "system", "user" ou "assistant"
    std::string contenu;
};

class Moteur {
public:
    // Charge un modèle GGUF. nullptr en cas d'échec (raison dans `erreur`).
    static std::unique_ptr<Moteur> ouvrir(const std::string & chemin, int n_ctx, int n_threads,
                                          std::string & erreur);
    ~Moteur();

    // Met la conversation au format du modèle (son propre modèle de conversation), réflexion désactivée.
    std::string formater(const std::vector<Message> & messages) const;

    // Génère la suite de `prompt`. `morceau` reçoit le texte au fur et à mesure (UTF-8 complet)
    // et renvoie false pour arrêter. Renvoie le texte généré ; `erreur` non vide en cas d'échec.
    std::string generer(const std::string & prompt, int max_tokens, float temperature,
                        const std::function<bool(const std::string &)> & morceau, std::string & erreur);

    std::string architecture() const;
    long long parametres() const;

private:
    Moteur() = default;
    llama_model * modele_ = nullptr;
    llama_context * contexte_ = nullptr;
    common_chat_templates * gabarits_ = nullptr;
    int n_batch_ = 512;
};

// Longueur du plus long préfixe de `s` fait de caractères UTF-8 complets.
size_t prefixe_utf8_complet(const std::string & s);

}  // namespace mirador
