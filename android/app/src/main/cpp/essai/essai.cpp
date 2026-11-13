// Essai du moteur sur PC : essai_moteur modele.gguf fichier_conversation.txt [max_tokens]
// Le fichier contient des blocs « ### role » suivis du texte du message.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "moteur.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage : essai_moteur modele.gguf conversation.txt [max_tokens]\n";
        return 2;
    }
    std::ifstream f(argv[2]);
    std::vector<mirador::Message> messages;
    std::string ligne;
    while (std::getline(f, ligne)) {
        if (ligne.rfind("### ", 0) == 0) {
            messages.push_back({ligne.substr(4), ""});
        } else if (!messages.empty()) {
            if (!messages.back().contenu.empty()) messages.back().contenu += "\n";
            messages.back().contenu += ligne;
        }
    }
    std::string erreur;
    auto t0 = std::chrono::steady_clock::now();
    auto moteur = mirador::Moteur::ouvrir(argv[1], 4096, getenv("FILS") ? atoi(getenv("FILS")) : 3, erreur);
    if (!moteur) { std::cerr << erreur << "\n"; return 1; }
    auto t1 = std::chrono::steady_clock::now();
    const std::string prompt = moteur->formater(messages);
    std::cerr << "[" << moteur->architecture() << ", " << moteur->parametres() / 1e6 << " M de paramètres, chargé en "
              << std::chrono::duration<double>(t1 - t0).count() << " s]\n";
    if (getenv("MONTRER_PROMPT")) std::cerr << "----- prompt -----\n" << prompt << "\n------------------\n";
    int n = 0;
    auto t2 = std::chrono::steady_clock::now();
    double premier = -1;
    const std::string texte = moteur->generer(prompt, argc > 3 ? atoi(argv[3]) : 400, 0.3f,
        [&](const std::string & morceau) {
            if (premier < 0) premier = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
            ++n; std::cout << morceau << std::flush; return true; }, erreur);
    auto t3 = std::chrono::steady_clock::now();
    const double duree = std::chrono::duration<double>(t3 - t2).count();
    std::cerr << "\n[" << n << " morceaux, premier après " << premier << " s, " << (n - 1) / std::max(1e-9, duree - premier)
              << " morceaux/s" << (erreur.empty() ? "" : ", erreur : " + erreur) << "]\n";
    return 0;
}
