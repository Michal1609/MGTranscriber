# Benchmark: čeština (FLEURS cs_cz, Common Voice, ParCzech4Speech)

Hlavní anglický benchmark je v [benchmark-en.md](benchmark-en.md).

Měřeno **2026-10-09 a 2026-10-10** na notebooku s **NVIDIA GeForce RTX 5070 Laptop GPU (8 GB)**, Windows 11,
Python 3.12, ONNX Runtime 1.24.4 (CUDA 12), PyTorch 2.8.0+cu128. Skript: [`benchmarks/bench_cs.py`](../benchmarks/bench_cs.py).
Všechny přepisy (300 na běh) a počty chyb jsou v [`benchmarks/results/`](../benchmarks/results/), tabulka se z nich
dá znovu vygenerovat: `uv run python -m benchmarks.bench_cs report benchmarks/results/*-n300.json`.

## Data a způsob měření

- Prvních 300 promluv testovací sady [FLEURS](https://huggingface.co/datasets/google/fleurs) `cs_cz`:
  59,8 minuty řeči, 5 586 slov, čtené věty z Wikipedie (licence CC-BY-4.0). Bez rozpoznávání mluvčích.
- **Model:** model se načte jednou a každou promluvu přepíše zvlášť (bez VAD). RTF nezahrnuje načtení modelu.
- **Aplikace:** všech 300 promluv se spojí do jedné nahrávky (1,5 s ticha mezi nimi) a projde
  `Engine.transcribe` přesně jako v aplikaci: VAD dělení, časy slov (Parakeet), zarovnání wav2vec2 (Canary).
  Slova se k promluvám přiřadí podle času. RTF zahrnuje i načtení modelu a zarovnání, je to doba, kterou uživatel čeká.
- Jazyk: Whisper a Canary dostanou `cs`. **Parakeet jazyk zadat nejde**, rozpozná ho sám.
- Normalizace pro referenci i hypotézu: malá písmena, bez interpunkce a symbolů. Čísla se nenormalizují,
  proto je zvlášť i WER na 236 promluvách, jejichž reference neobsahuje číslici.
- WER a CER jsou korpusové (součet chyb / součet slov). Interval je 95% bootstrap přes promluvy (2 000 vzorků),
  Δ je párový bootstrap rozdílu proti nejlepšímu běhu na stejných promluvách.

## Výsledky

| běh | WER % (95% interval) | chyby / slova | WER % bez číslic | CER % | Δ WER proti Canary (95% interval) | RTF | rychlost |
|---|---|---|---|---|---|---|---|
| Canary-1B-v2, model | **8,97** (8,0–9,9) | 501 / 5 586 | **8,52** | **3,02** | – | 0,088 | 11× realtime |
| Canary-1B-v2, aplikace | 9,56 (8,6–10,5) | 534 / 5 586 | 9,13 | 3,19 | +0,59 (−0,02…+1,21) | 0,086 | 12× |
| Whisper large-v3, model | 10,49 (9,4–11,5) | 586 / 5 586 | 10,28 | 3,08 | +1,52 (+0,66…+2,42) | 0,122 | 8× |
| Parakeet-TDT-0.6B-v3, aplikace | 10,54 (9,5–11,6) | 589 / 5 586 | 10,19 | 3,03 | +1,58 (+0,68…+2,49) | **0,022** | **45×** |
| Whisper large-v3-turbo, model | 10,85 (9,9–11,9) | 606 / 5 586 | 10,31 | 3,09 | +1,88 (+1,01…+2,75) | 0,034 | 29× |
| Parakeet-TDT-0.6B-v3, model | 11,37 (10,2–12,5) | 635 / 5 586 | 11,11 | 3,22 | +2,40 (+1,36…+3,46) | **0,022** | 46× |

Načtení modelu (běh „model“): Parakeet 6 s, Canary 9 s, turbo 19 s, large-v3 33 s.
Běh „aplikace“ trval celkem 90 s (Parakeet) a 347 s (Canary, z toho zarovnání slov) na 67 minut nahrávky (59,8 min promluv a 7,5 min ticha mezi nimi).

## Celý přepis s mluvčími (88 minut, angličtina)

Rychlost výše je samotné rozpoznání řeči. Uživatel čeká na celou cestu aplikace včetně rozpoznání mluvčích
(pyannote community-1). Změřeno na 87,6 minuty anglického rozhovoru se dvěma mluvčími, nahrávka nahraná
přes web, RTX 5070 Laptop, `--speakers 2`, jeden běh každého jádra:

| jádro | celý přepis | z toho rozpoznání a zarovnání slov | z toho mluvčí |
|---|---|---|---|
| Whisper large-v3-turbo (WhisperX) | 5 min 55 s | ≈ 2,9 min | ≈ 3,1 min |
| Parakeet-TDT-0.6B-v3 | 6 min 27 s | ≈ 3,2 min | ≈ 3,3 min |
| Canary-1B-v2 | 7 min 48 s | ≈ 4,7 min | ≈ 3,1 min |

Čas mluvčích je u všech jader zhruba stejný (3,1–3,3 min) a je největší jednotlivá položka, proto
Parakeet na celém přepisu není znatelně rychlejší než turbo (čísla jsou z jednoho běhu, rozdíly pár desítek
sekund nejsou průkazné). Časy rozdělení jsou odečtené z časových značek v logu úlohy, ne samostatně změřené.

## Parlamentní řeč s mluvčími (ParCzech4Speech, 88 minut)

Měřeno 2026-10-10 na stejném notebooku, přes aplikaci, **s rozpoznáním mluvčích a bez udání jejich počtu**,
jeden běh každého jádra. Skripty: `--dataset parczech` v [`benchmarks/bench_cs.py`](../benchmarks/bench_cs.py),
sestavení relací v [`benchmarks/longform.py`](../benchmarks/longform.py). Anglický protějšek (porady AMI) je v
[benchmark-en.md](benchmark-en.md).

**Data:** testovací sada [ParCzech4Speech 1.0](https://lindat.mff.cuni.cz/repository/items/8cc95ecb-680f-4454-b041-58a787a5ca5f)
(Poslanecká sněmovna, licence CC-BY; Stankov, Kopp, Bojar, *ParCzech4Speech*, TSD 2025, arXiv:2509.06675;
odvozeno z ParCzech 4.0 a AudioPSP 24.01). Devět relací po ~10 minutách (87,6 min, 9 668 referenčních slov,
43 různých mluvčích, 4–6 v relaci) ze 3 volebních období. Referencí je oficiální (stenografický) přepis.

| běh | WER % na klip (95% interval) | WER % celé relace | Δ WER proti large-v3 | DER % | miss / FA / záměna % | mluvčí odhad / skutečně | celý přepis | rychlost |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3 | **8,09** (7,3–8,8) | **7,89** | – | 4,27 | 3,9 / 0,1 / 0,3 | 43 / 43 | 12,3 min | 7× |
| Canary-1B-v2 | 8,10 (7,4–8,8) | **7,89** | +0,01 (−0,76…+0,74) | 3,50 | 3,1 / 0,1 / 0,3 | 43 / 43 | 12,5 min | 7× |
| Whisper large-v3-turbo | 8,81 (7,9–9,8) | 8,61 | +0,72 (+0,03…+1,49) | 4,72 | 4,2 / 0,1 / 0,4 | 43 / 43 | 8,9 min | 10× |
| Parakeet-TDT-0.6B-v3 | 10,05 (9,2–10,8) | 9,66 | +1,97 (+1,16…+2,73) | **2,81** | 2,3 / 0,1 / 0,4 | 43 / 43 | **7,8 min** | **11×** |

Postup: každá relace je jedna nahrávka (klipy jednoho zasedání seřazené podle pořadí v zasedání, 0,4 s ticha
mezi nimi) a prochází `Engine.transcribe` s `diarize=True`. WER „na klip“ přiřadí slova ke klipům podle času
(interval je bootstrap přes 591 klipů); „celé relace“ je editační vzdálenost celého textu relace. DER je
z pyannote.metrics, kolárek 0,25 s, počítaný z časů a mluvčích slov ve výstupu aplikace (viz benchmark-en.md).
Normalizace je stejná jako u FLEURS (čísla se nenormalizují).

Co z toho plyne:

- **Na přepisu vede large-v3 a Canary**, Parakeet je o 2 body horší (interval nad nulou) a turbo o 0,7. Parakeet
  je ale nejrychlejší a má nejnižší DER, nejspíš proto, že přepisuje i krátké úseky, které ostatní jádra
  vynechají (na češtině jsme to neověřovali; u angličtiny viz benchmark-en.md).
- **Mluvčí vyšli bezchybně v počtu (43 ze 43) a s 0,3–0,4 % záměn.** To je ale optimistické, viz omezení.
- Celý přepis 88 minut trvá 8–13 minut (7–11× reálný čas) včetně načtení modelu u každé relace.

Omezení tohoto měření (důležitá):

- **Relace jsou poskládané, ne souvislé.** ParCzech4Speech jsou samostatné jednomluvčí klipy se zarovnáním,
  bez absolutních časů a bez překryvů. Relace je mozaika z klipů jednoho zasedání: mluvčí se střídají po celých
  projevech, nikdy nemluví přes sebe a mezi nimi jsou čisté pauzy. DER je proto **optimistický** a hodí se hlavně
  k porovnání jader, ne jako odhad na skutečné jednání. Souvislé nahrávky (AudioPSP) jsme nestahovali.
- **Reference je stenografický text**, ne doslovný přepis: vynechává přeřeknutí a zkracuje, takže je WER u všech
  vyšší, než jaký by byl proti doslovné referenci. Zarovnání textu na zvuk dělal WhisperX s wav2vec2, což může
  mírně zvýhodňovat jádra s Whisperem. Rozdíly mezi jádry proto čtěte opatrně.
- DER ve sloupci „miss“ zahrnuje i ticho na začátku a konci klipu, které reference počítá jako řeč.
Licence: ParCzech4Speech je CC-BY; texty referencí a hypotéz se v `benchmarks/results/*-parczech-*.json` nezveřejňují (jen id, metriky a počty chyb), audio se
nerozšiřuje (stahuje ho skript).

## Common Voice cs (nahrávky dobrovolníků, 27.0)

Změřeno **2026-10-10**, stejný notebook i postup jako výše, všechny běhy cestou aplikace (`--app`, bez mluvčích, GPU):
`uv run python -m benchmarks.bench_cs --dataset cv -n 300 --app whisperx:large-v3 --app whisperx:large-v3-turbo --app parakeet --app canary`.

- **Data:** testovací část Common Voice Corpus 27.0 (2026-09-11, `cs`): 9 493 vět od 552 mluvčích (klientů).
  Vybraných 300 klipů: pevný seed 0, nejvýše 3 klipy od jednoho mluvčího (185 mluvčích), každá věta jednou.
  2 265 slov, v průměru 7,5 slova na klip; spojená nahrávka trvá 31,2 min včetně pauz 1,5 s.
  Klipy jsou mp3 nahrané doma na různé mikrofony, tedy horší a nesourodější zvuk než FLEURS.
- Normalizace, WER, CER a intervaly stejně jako výše. Reference žádnou číslici neobsahuje, takže „WER bez číslic“
  je shodný s WER.

| běh (aplikace) | WER % (95% interval) | chyby / slova | CER % | Δ WER proti Parakeetu (95% interval) | RTF | rychlost |
|---|---|---|---|---|---|---|
| Parakeet-TDT-0.6B-v3 | **9,93** (8,1–11,9) | 225 / 2 265 | 3,15 | – | **0,026** | **38×** |
| Whisper large-v3-turbo | 10,91 (9,0–13,1) | 247 / 2 265 | 3,31 | +0,97 (−1,12…+3,18) | 0,036 | 28× |
| Whisper large-v3 | 11,48 (9,5–13,5) | 260 / 2 265 | 4,00 | +1,55 (−0,59…+3,73) | 0,096 | 10× |
| Canary-1B-v2 | 11,74 (9,8–13,9) | 266 / 2 265 | 3,44 | +1,81 (−0,05…+3,74) | 0,067 | 15× |

Co z toho plyne:

- **Rozdíly mezi modely nejsou prokazatelné.** Všechny párové intervaly zahrnují nulu a intervaly WER se
  výrazně překrývají; 300 krátkých klipů (2 265 slov) dává interval široký asi ±2 body. Na těchto datech nejde
  říct, že je některý model přesnější. Bodově je nejlepší Parakeet, nejhorší Canary.
- **Pořadí se oproti FLEURS (Canary první) liší**, ale protože rozdíly jsou v rámci šumu, nejde to číst jako
  změnu závěru: čtená FLEURS a domácí nahrávky mají jiný zvuk i jiné věty. Společné je jen to, že všechny čtyři
  běhy leží blízko u sebe (9,9–11,7 %).
- Celý běh trval 50 s (Parakeet), 68 s (turbo), 126 s (Canary) a 179 s (large-v3) na 31 min nahrávky.

Licence a podmínky: Common Voice je CC0, ale data se stahují z Mozilla Data Collective po přihlášení a souhlasu
s jeho podmínkami, které zakazují další zveřejňování dat. Proto jsou v
`benchmarks/results/*-cv-app-*.json` jen id klipů, metriky a počty chyb po větách, **bez referenčních vět i hypotéz**, a audio ani
`test.tsv` se v repozitáři nenacházejí (`benchmarks/data/` je ignorováno). Data se musí stáhnout a rozbalit ručně do
`benchmarks/data/cv_cs/` (`test.tsv` + `clips/`).

## Co z toho plyne

- **Canary je na čtené češtině (FLEURS) nejpřesnější** (na domácích nahrávkách Common Voice se to nepotvrdilo, tam jsou modely nerozlišitelné, viz výše). Proti Whisperu large-v3 je o 1,5 bodu WER lepší a interval
  rozdílu (+0,66…+2,42) nule nepřekrývá. Platí to i bez promluv s číslicemi (8,52 proti 10,28 %), takže to
  není jen tím, že Canary píše čísla číslicemi. CER mají všechny modely skoro stejné (3,0–3,2 %): rozdíly jsou
  hlavně v koncovkách a tvarech slov, ne ve „slyšení“.
- **Whisper large-v3, turbo a Parakeet jsou statisticky nerozlišitelné.** Párové rozdíly proti large-v3:
  turbo +0,36 (−0,38…+1,08), Parakeet (model) +0,88 (−0,22…+1,98), Parakeet (aplikace) +0,05 (−0,90…+1,01).
  Na 300 větách nejde říct, který z nich je přesnější. Publikovaná čísla (NVIDIA, arXiv 2509.14128: large-v3
  11,33 %, Parakeet 11,01 %, Canary 7,86 %) ukazují stejné pořadí.
- **Rychlost rozpoznání:** Parakeet je nejrychlejší (45× realtime i s celou cestou aplikace bez mluvčích), Canary 12×,
  turbo 29×. S rozpoznáním mluvčích se rozdíl v celkovém čase skoro ztratí, viz tabulka výše. Large-v3 jsme měřili jen po promluvách; v aplikaci WhisperX dávkuje, takže tam bude rychlejší.
- **Parakeet a jazyk:** model jazyk nedostane, v jednotlivých větách proto občas sklouzne do slovenštiny
  („v tomto ohľadu … podobne ako“). V cestě aplikace s delšími úseky se to na těchto datech neobjevilo
  (kontrola jazyka v aplikaci nehlásila nic), ale na spontánní řeči nebo s cizími slovy to vyloučit nejde.
  Aplikace v takovém případě zapíše do logu varování.

## Chyba nalezená benchmarkem

První běh přes aplikaci dopadl špatně: Parakeet 17,70 % (16,0–19,4) a Canary 13,01 % (11,8–14,3). onnx-asr
standardně dělí řeč v každé pauze delší než 100 ms. Krátké úseky daly Parakeetu tak málo kontextu, že část
českých vět přepsal ukrajinsky, rusky nebo polsky („Витворила білоруська українська фронти“), a Canary ztrácel
souvislost. Aplikace teď nechává pauzy do 2 s uvnitř úseku dlouhého až 20 s (`MIN_SILENCE_MS`). Oba původní
běhy jsou pro srovnání v `benchmarks/results/*-vad100ms-*.json`. Kontrola jazyka po úsecích teď takové
přepisy pozná (na prvním běhu Parakeetu hlásí 42 ze 686 úseků).

## Omezení

- FLEURS jsou čtené věty v dobré kvalitě. Na spontánní řeči, rozhovorech a horším zvuku budou všechny modely
  horší a pořadí se může lišit. Parlamentní řeč s mluvčími je změřená výše (s vlastními omezeními), spontánní
  rozhovory na českých datech zatím ne.
- Jeden stroj, sdílený s dalšími procesy (paměť RAM byla během měření chvílemi téměř plná), takže časy mohou
  být mírně horší než na volném stroji. WER to neovlivňuje.
- Přiřazení slov k promluvám podle času (běh „aplikace“) může slovo na hranici dát sousední promluvě. Při 1,5 s
  ticha mezi promluvami je to vzácné, ale WER tohoto běhu je tím spíš mírně nadhodnocené.
