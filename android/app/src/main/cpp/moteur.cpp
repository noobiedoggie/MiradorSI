#include "moteur.h"

#include <algorithm>
#include <ctime>

#include "chat.h"
#include "llama.h"

namespace mirador {

namespace {
void journal_silencieux(ggml_log_level niveau, const char * texte, void *) {
    if (niveau >= GGML_LOG_LEVEL_ERROR) fputs(texte, stderr);
}
}  // namespace

size_t prefixe_utf8_complet(const std::string & s) {
    const size_t n = s.size();
    size_t i = n;
    int suite = 0;  // octets de continuation (10xxxxxx) en fin de chaîne
    while (i > 0 && suite < 3 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) { --i; ++suite; }
    if (i == 0) return n;
    const unsigned char c = static_cast<unsigned char>(s[i - 1]);
    const size_t debut = i - 1;
    const size_t attendu = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    return (n - debut >= attendu) ? n : debut;
}

std::unique_ptr<Moteur> Moteur::ouvrir(const std::string & chemin, int n_ctx, int n_threads, std::string & erreur) {
    static bool initialise = false;
    if (!initialise) {
        llama_log_set(journal_silencieux, nullptr);
        llama_backend_init();
        initialise = true;
    }
    std::unique_ptr<Moteur> m(new Moteur());
    auto params = llama_model_default_params();
    m->modele_ = llama_model_load_from_file(chemin.c_str(), params);
    if (!m->modele_) {
        erreur = "impossible de charger le modèle : " + chemin;
        return nullptr;
    }
    auto cparams = llama_context_default_params();
    const int entraine = llama_model_n_ctx_train(m->modele_);
    cparams.n_ctx = static_cast<uint32_t>(entraine > 0 ? std::min(n_ctx, entraine) : n_ctx);
    cparams.n_batch = static_cast<uint32_t>(m->n_batch_);
    cparams.n_ubatch = static_cast<uint32_t>(m->n_batch_);
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    m->contexte_ = llama_init_from_model(m->modele_, cparams);
    if (!m->contexte_) {
        erreur = "mémoire insuffisante pour ce modèle";
        return nullptr;
    }
    m->gabarits_ = common_chat_templates_init(m->modele_, "").release();
    return m;
}

Moteur::~Moteur() {
    if (gabarits_) common_chat_templates_free(gabarits_);
    if (contexte_) llama_free(contexte_);
    if (modele_) llama_model_free(modele_);
}

std::string Moteur::formater(const std::vector<Message> & messages) const {
    common_chat_templates_inputs entrees;
    for (const auto & m : messages) {
        common_chat_msg msg;
        msg.role = m.role;
        msg.content = m.contenu;
        entrees.messages.push_back(msg);
    }
    entrees.add_generation_prompt = true;
    entrees.use_jinja = true;
    entrees.enable_thinking = false;  // réponse directe, sans « réflexion » cachée
    return common_chat_templates_apply(gabarits_, entrees).prompt;
}

std::string Moteur::generer(const std::string & prompt, int max_tokens, float temperature,
                            const std::function<bool(const std::string &)> & morceau, std::string & erreur) {
    const llama_vocab * vocab = llama_model_get_vocab(modele_);
    llama_memory_clear(llama_get_memory(contexte_), true);  // chaque question repart de zéro

    const int n = -llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    std::vector<llama_token> jetons(static_cast<size_t>(std::max(n, 0)));
    if (llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), jetons.data(),
                       static_cast<int32_t>(jetons.size()), true, true) < 0) {
        erreur = "découpage du texte impossible";
        return "";
    }
    const int n_ctx = static_cast<int>(llama_n_ctx(contexte_));
    if (static_cast<int>(jetons.size()) + 8 > n_ctx) {  // trop long : on garde la fin (la question)
        jetons.erase(jetons.begin(), jetons.end() - (n_ctx - 8 - std::min(max_tokens, n_ctx / 4)));
    }
    for (size_t i = 0; i < jetons.size(); i += static_cast<size_t>(n_batch_)) {
        const int taille = static_cast<int>(std::min(jetons.size() - i, static_cast<size_t>(n_batch_)));
        if (llama_decode(contexte_, llama_batch_get_one(jetons.data() + i, taille)) != 0) {
            erreur = "lecture de la question impossible";
            return "";
        }
    }

    llama_sampler * echantillon = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(echantillon, llama_sampler_init_top_k(40));  // avant les pénalités : plus rapide
    llama_sampler_chain_add(echantillon,
                            llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1f, 0.0f, 0.0f));
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(echantillon, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(echantillon, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(echantillon, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(echantillon, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(echantillon, llama_sampler_init_dist(static_cast<uint32_t>(time(nullptr))));
    }

    std::string sortie, attente;
    int position = static_cast<int>(jetons.size());
    for (int i = 0; i < max_tokens && position < n_ctx; ++i, ++position) {
        llama_token suivant = llama_sampler_sample(echantillon, contexte_, -1);
        if (llama_vocab_is_eog(vocab, suivant)) break;
        char tampon[256];
        const int lu = llama_token_to_piece(vocab, suivant, tampon, sizeof(tampon), 0, false);
        if (lu > 0) {
            attente.append(tampon, static_cast<size_t>(lu));
            const size_t complet = prefixe_utf8_complet(attente);
            if (complet > 0) {
                const std::string texte = attente.substr(0, complet);
                attente.erase(0, complet);
                sortie += texte;
                if (morceau && !morceau(texte)) break;
            }
        }
        if (llama_decode(contexte_, llama_batch_get_one(&suivant, 1)) != 0) {
            erreur = "génération interrompue";
            break;
        }
    }
    llama_sampler_free(echantillon);
    return sortie;
}

std::string Moteur::architecture() const {
    char tampon[128] = {0};
    llama_model_meta_val_str(modele_, "general.architecture", tampon, sizeof(tampon));
    return tampon;
}

long long Moteur::parametres() const {
    return static_cast<long long>(llama_model_n_params(modele_));
}

}  // namespace mirador
