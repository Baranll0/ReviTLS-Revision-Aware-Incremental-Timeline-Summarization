# Synthetic Revision Benchmark Catalog (32 Scenarios)

Total scenarios documented: **32**

| Suite | Topic | Case ID | Type | Delay (d) | Target Status | Expected Final Fact | Forbidden Stale Substring |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| 16-Scenario Multi-Topic Suite | `bpoil` | `rev_numeric_spill` | `NUMERIC_FACT` | 6 | `REVISED` | 5,000 barrels | `1,000 barrels` |
| 16-Scenario Multi-Topic Suite | `bpoil` | `rev_date_shift` | `DATE_CORRECTION` | 4 | `REVISED` | {corrected_date} | `{initial_date}` |
| 16-Scenario Multi-Topic Suite | `bpoil` | `rev_entity_attribution` | `ENTITY_ATTRIBUTION` | 3 | `REVISED` | valve maintenance error | `software glitch` |
| 16-Scenario Multi-Topic Suite | `bpoil` | `rev_retraction_rumor` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `secondary pipeline rupture` |
| 16-Scenario Multi-Topic Suite | `finan` | `finan_numeric_shortfall` | `NUMERIC_FACT` | 7 | `REVISED` | $28 billion | `$12 billion` |
| 16-Scenario Multi-Topic Suite | `finan` | `finan_date_margin` | `DATE_CORRECTION` | 5 | `REVISED` | {corrected_date} | `{initial_date}` |
| 16-Scenario Multi-Topic Suite | `finan` | `finan_entity_cause` | `ENTITY_ATTRIBUTION` | 4 | `REVISED` | unauthorized derivatives desk | `trading bug` |
| 16-Scenario Multi-Topic Suite | `finan` | `finan_retraction_freeze` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `ATM cash withdrawal suspension` |
| 16-Scenario Multi-Topic Suite | `h1n1` | `h1n1_numeric_cases` | `NUMERIC_FACT` | 8 | `REVISED` | 1,850 | `420` |
| 16-Scenario Multi-Topic Suite | `h1n1` | `h1n1_date_onset` | `DATE_CORRECTION` | 4 | `REVISED` | {corrected_date} | `{initial_date}` |
| 16-Scenario Multi-Topic Suite | `h1n1` | `h1n1_entity_vector` | `ENTITY_ATTRIBUTION` | 5 | `REVISED` | reassortant swine strain | `avian vector` |
| 16-Scenario Multi-Topic Suite | `h1n1` | `h1n1_retraction_border` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `airport quarantine closure` |
| 16-Scenario Multi-Topic Suite | `haiti` | `haiti_numeric_shelters` | `NUMERIC_FACT` | 6 | `REVISED` | 48,000 | `15,000` |
| 16-Scenario Multi-Topic Suite | `haiti` | `haiti_date_convoy` | `DATE_CORRECTION` | 3 | `REVISED` | {corrected_date} | `{initial_date}` |
| 16-Scenario Multi-Topic Suite | `haiti` | `haiti_entity_bridge` | `ENTITY_ATTRIBUTION` | 4 | `REVISED` | pier liquefaction failure | `hillside landslide` |
| 16-Scenario Multi-Topic Suite | `haiti` | `haiti_retraction_tsunami` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `secondary tsunami wave` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_fict_numeric_bridge` | `NUMERIC_FACT` | 5 | `REVISED` | 62 | `14 hairline` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_fict_date_kestrel` | `DATE_CORRECTION` | 4 | `REVISED` | {corrected_date} | `{initial_date}` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_fict_cause_oakhaven` | `ENTITY_ATTRIBUTION` | 4 | `REVISED` | baffle rupture | `algal bloom` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_fict_retract_solstice` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `satellite breakup` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_real_numeric_macondo` | `NUMERIC_FACT` | 6 | `REVISED` | 8,350 barrels | `2,400 barrels` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_real_cause_rig` | `ENTITY_ATTRIBUTION` | 4 | `REVISED` | fuel-line fracture | `generator overload` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_real_numeric_lehman` | `NUMERIC_FACT` | 5 | `REVISED` | $63 billion | `$18 billion` |
| 8-Scenario Parametric Control | `counterfactual_control` | `cf_real_retract_fourchon` | `RETRACTION` | 2 | `RETRACTED` | *(excluded)* | `subsea storage tank explosion` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_num_reservoir` | `NUMERIC_FACT` | 18 | `REVISED` | 34,000 cubic meters | `12,000 cubic meters` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_num_casualties` | `NUMERIC_FACT` | 25 | `REVISED` | 29 dockworkers | `14 injured` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_causal_grid` | `ENTITY_ATTRIBUTION` | 15 | `REVISED` | corroded relay switch | `lightning strike` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_causal_cargo` | `ENTITY_ATTRIBUTION` | 30 | `REVISED` | lithium battery | `diesel fuel line` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_date_quake` | `DATE_CORRECTION` | 21 | `REVISED` | {corrected_date} | `{initial_date}` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_date_treaty` | `DATE_CORRECTION` | 35 | `REVISED` | {corrected_date} | `{initial_date}` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_retract_bridge` | `RETRACTION` | 16 | `RETRACTED` | *(excluded)* | `girder crack` |
| 8-Scenario Implicit & Long-Delay | `implicit_long_delay` | `imp_retract_vaccine` | `RETRACTION` | 28 | `RETRACTED` | *(excluded)* | `cold-chain spoilage` |

## Full Scenario Texts (Initial Claim vs. Corrective Evidence)

### 1. [16-Scenario Multi-Topic Suite] `bpoil` / `rev_numeric_spill` (`REVISED`, $\Delta t=6$d)
- **Initial (3 repeats):** *Coast Guard estimates leak at 1,000 barrels per day* — Officials announced today that the damaged offshore Macondo well is leaking approximately 1,000 barrels of crude oil per day into the Gulf.
- **Correction:** *Scientists revise offshore leak estimate upward to 5,000 barrels per day* — Correction and updated assessment: Federal scientists revised the estimate of the Macondo well leak upward to 5,000 barrels of oil per day, superseding earlier 1,000-barrel estimates.
- **Evaluation Rule:** Target status `REVISED`, expected=`5,000 barrels`, forbidden=`1,000 barrels`

### 2. [16-Scenario Multi-Topic Suite] `bpoil` / `rev_date_shift` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Regulatory hearing reportedly held on initial collapse* — Early press dispatches reported that the emergency regulatory panel convened on {initial_date} to question bank executives over liquidity failures.
- **Correction:** *Clarification: Senate Banking hearing actually occurred on {corrected_date}* — Editor's note correcting earlier schedule reports: The formal Senate Banking Committee hearing on the bank failure took place on {corrected_date}, not {initial_date} as previously stated.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 3. [16-Scenario Multi-Topic Suite] `bpoil` / `rev_entity_attribution` (`REVISED`, $\Delta t=3$d)
- **Initial (2 repeats):** *Initial probe blames software glitch for plant outage* — Preliminary statements suggested an automated software glitch triggered the emergency shutdown at the Meridian chemical plant.
- **Correction:** *Investigators rule out software glitch, cite valve maintenance error* — Official inquiry update: Safety investigators retracted the software glitch theory and confirmed a manual valve maintenance error caused the Meridian plant outage.
- **Evaluation Rule:** Target status `REVISED`, expected=`valve maintenance error`, forbidden=`software glitch`

### 4. [16-Scenario Multi-Topic Suite] `bpoil` / `rev_retraction_rumor` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of secondary pipeline rupture at Harbor Terminal* — Local social channels claimed a secondary pipeline rupture occurred at Harbor Terminal following the main incident.
- **Correction:** *Authorities officially debunk Harbor Terminal pipeline rupture rumor* — Official retraction: Coast Guard and terminal operators confirmed no secondary pipeline rupture occurred at Harbor Terminal; earlier reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`secondary pipeline rupture`

### 5. [16-Scenario Multi-Topic Suite] `finan` / `finan_numeric_shortfall` (`REVISED`, $\Delta t=7$d)
- **Initial (3 repeats):** *Analysts estimate regional bank liquidity shortfall at $12 billion* — Initial market filings indicated that Northstar Commercial Bank faced a liquidity shortfall of $12 billion following mortgage-backed asset write-downs.
- **Correction:** *Audit revises Northstar Bank liquidity shortfall upward to $28 billion* — Correction and audited filing: External auditors revised Northstar Commercial Bank's liquidity shortfall upward to $28 billion, superseding the preliminary $12 billion estimate.
- **Evaluation Rule:** Target status `REVISED`, expected=`$28 billion`, forbidden=`$12 billion`

### 6. [16-Scenario Multi-Topic Suite] `finan` / `finan_date_margin` (`REVISED`, $\Delta t=5$d)
- **Initial (2 repeats):** *Emergency clearinghouse margin call reportedly issued on {initial_date}* — Early wire reports stated that the sovereign bond clearinghouse issued an emergency margin call on {initial_date} to cover overnight repo exposure.
- **Correction:** *Clarification: Clearinghouse margin call actually occurred on {corrected_date}* — Editor's note correcting earlier timeline: Market regulators confirmed the emergency clearinghouse margin call took place on {corrected_date}, not {initial_date} as previously stated.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 7. [16-Scenario Multi-Topic Suite] `finan` / `finan_entity_cause` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Preliminary review blames automated trading bug for fund freeze* — Initial spokespersons suggested an automated algorithmic trading bug triggered the sudden redemption freeze at Crestline Capital.
- **Correction:** *Regulators rule out trading bug, cite unauthorized derivatives desk* — Official enforcement update: Financial investigators retracted the automated trading bug theory and confirmed an unauthorized derivatives desk caused the Crestline Capital freeze.
- **Evaluation Rule:** Target status `REVISED`, expected=`unauthorized derivatives desk`, forbidden=`trading bug`

### 8. [16-Scenario Multi-Topic Suite] `finan` / `finan_retraction_freeze` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of nationwide ATM cash withdrawal suspension* — Social feeds claimed federal regulators ordered a nationwide ATM cash withdrawal suspension across retail branches.
- **Correction:** *Central bank officially debunks nationwide ATM suspension rumor* — Official retraction: The central bank confirmed no nationwide ATM cash withdrawal suspension occurred; earlier reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`ATM cash withdrawal suspension`

### 9. [16-Scenario Multi-Topic Suite] `h1n1` / `h1n1_numeric_cases` (`REVISED`, $\Delta t=8$d)
- **Initial (3 repeats):** *Health ministry reports 420 confirmed respiratory flu cases* — Provincial health bulletins initially reported 420 laboratory-confirmed cases of the novel influenza strain across regional clinics.
- **Correction:** *Surveillance update revises confirmed flu cases upward to 1,850* — Correction and updated epidemiology report: National health labs revised the confirmed novel influenza count upward to 1,850 cases, superseding the earlier 420-case figure.
- **Evaluation Rule:** Target status `REVISED`, expected=`1,850`, forbidden=`420`

### 10. [16-Scenario Multi-Topic Suite] `h1n1` / `h1n1_date_onset` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *First cross-border cluster reportedly detected on {initial_date}* — Initial dispatches stated that the first cross-border hospital cluster was detected on {initial_date} at the northern transit hub.
- **Correction:** *Clarification: First cross-border cluster actually occurred on {corrected_date}* — Correction note: Epidemiologists confirmed the first cross-border hospital cluster took place on {corrected_date}, not {initial_date} as previously stated.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 11. [16-Scenario Multi-Topic Suite] `h1n1` / `h1n1_entity_vector` (`REVISED`, $\Delta t=5$d)
- **Initial (2 repeats):** *Initial lab brief blames migratory avian vector for outbreak* — Early virology notes suggested a migratory avian vector introduced the respiratory cluster near the border valley.
- **Correction:** *Genomic sequencing rules out avian vector, cites swine reassortant* — Official sequencing update: Reference labs retracted the migratory avian vector hypothesis and confirmed a triple-reassortant swine strain caused the outbreak.
- **Evaluation Rule:** Target status `REVISED`, expected=`reassortant swine strain`, forbidden=`avian vector`

### 12. [16-Scenario Multi-Topic Suite] `h1n1` / `h1n1_retraction_border` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of mandatory international airport quarantine closure* — Online posts claimed aviation authorities imposed a mandatory quarantine closure at all international arrival terminals.
- **Correction:** *Aviation ministry debunks international airport quarantine closure rumor* — Official retraction: Aviation and health officials confirmed no mandatory international airport quarantine closure occurred; earlier reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`airport quarantine closure`

### 13. [16-Scenario Multi-Topic Suite] `haiti` / `haiti_numeric_shelters` (`REVISED`, $\Delta t=6$d)
- **Initial (3 repeats):** *Relief agencies count 15,000 displaced residents in port camps* — Initial field surveys estimated 15,000 displaced residents sheltering in temporary encampments near the damaged coastal port.
- **Correction:** *UN assessment revises displaced camp population upward to 48,000* — Correction and revised census: UN relief teams revised the displaced camp population upward to 48,000 residents, superseding earlier 15,000 estimates.
- **Evaluation Rule:** Target status `REVISED`, expected=`48,000`, forbidden=`15,000`

### 14. [16-Scenario Multi-Topic Suite] `haiti` / `haiti_date_convoy` (`REVISED`, $\Delta t=3$d)
- **Initial (2 repeats):** *Emergency medical airlift reportedly landed on {initial_date}* — Early dispatches reported that the first heavy-lift medical supply convoy landed on {initial_date} at the southern airstrip.
- **Correction:** *Clarification: Heavy-lift medical convoy actually landed on {corrected_date}* — Correction note: Logistics coordinators confirmed the heavy-lift medical supply convoy took place on {corrected_date}, not {initial_date} as previously stated.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 15. [16-Scenario Multi-Topic Suite] `haiti` / `haiti_entity_bridge` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Initial survey blames hillside landslide for highway bridge collapse* — Preliminary road reports suggested a hillside landslide severed the main supply highway bridge outside the capital.
- **Correction:** *Engineers rule out landslide, cite pier liquefaction failure* — Official engineering update: Structural teams retracted the hillside landslide theory and confirmed coastal pier liquefaction failure caused the bridge collapse.
- **Evaluation Rule:** Target status `REVISED`, expected=`pier liquefaction failure`, forbidden=`hillside landslide`

### 16. [16-Scenario Multi-Topic Suite] `haiti` / `haiti_retraction_tsunami` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of destructive secondary tsunami wave hitting bay* — Local radio rumors claimed a destructive secondary tsunami wave struck the western harbor three hours after the tremor.
- **Correction:** *Tsunami center officially debunks secondary harbor wave rumor* — Official retraction: Coastal monitoring stations confirmed no secondary tsunami wave occurred in the western harbor; earlier reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`secondary tsunami wave`

### 17. [8-Scenario Parametric Control] `counterfactual_control` / `cf_fict_numeric_bridge` (`REVISED`, $\Delta t=5$d)
- **Initial (3 repeats):** *Inspectors log 14 hairline fractures on Vespera Rail Bridge* — Municipal engineers initially reported 14 hairline fractures along the main cantilever span of the Vespera Rail Bridge.
- **Correction:** *Ultrasonic scan revises Vespera Bridge fracture count to 62* — Correction and ultrasonic audit: Structural inspectors revised the Vespera Rail Bridge count upward to 62 structural fractures, superseding earlier 14-fracture reports.
- **Evaluation Rule:** Target status `REVISED`, expected=`62`, forbidden=`14 hairline`

### 18. [8-Scenario Parametric Control] `counterfactual_control` / `cf_fict_date_kestrel` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Kestrel Biotech Phase-III trial readout reportedly held on {initial_date}* — Early press notes stated that Kestrel Biotech convened its Phase-III clinical review panel on {initial_date}.
- **Correction:** *Clarification: Kestrel Biotech panel actually met on {corrected_date}* — Editor's note correcting earlier schedule: Kestrel Biotech confirmed the Phase-III panel took place on {corrected_date}, not {initial_date} as previously stated.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 19. [8-Scenario Parametric Control] `counterfactual_control` / `cf_fict_cause_oakhaven` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Initial water test blames seasonal algal bloom for Oakhaven turbidity* — Preliminary utility statements suggested a seasonal algal bloom caused the turbidity spike at the Oakhaven Municipal Reservoir.
- **Correction:** *Divers rule out algal bloom, cite submerged baffle rupture* — Official utility correction: Inspection divers retracted the algal bloom theory and confirmed a submerged intake baffle rupture caused the Oakhaven turbidity spike.
- **Evaluation Rule:** Target status `REVISED`, expected=`baffle rupture`, forbidden=`algal bloom`

### 20. [8-Scenario Parametric Control] `counterfactual_control` / `cf_fict_retract_solstice` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of Solstice Orbital Telemetry satellite breakup* — Unverified forum posts claimed a debris strike caused the breakup of the Solstice Orbital Telemetry relay satellite.
- **Correction:** *Space agency debunks Solstice Orbital Telemetry breakup rumor* — Official retraction: Ground tracking stations confirmed no debris strike or breakup occurred on the Solstice Orbital Telemetry satellite; reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`satellite breakup`

### 21. [8-Scenario Parametric Control] `counterfactual_control` / `cf_real_numeric_macondo` (`REVISED`, $\Delta t=6$d)
- **Initial (3 repeats):** *Early gauge estimates Macondo flow at 2,400 barrels per day* — Preliminary pressure logs estimated the offshore Macondo well was discharging 2,400 barrels of crude oil per day.
- **Correction:** *Revised sonar metering puts Macondo flow at 8,350 barrels per day* — Correction and updated telemetry: Acoustic flow metering revised the Macondo well discharge upward to 8,350 barrels per day, superseding earlier 2,400-barrel figures.
- **Evaluation Rule:** Target status `REVISED`, expected=`8,350 barrels`, forbidden=`2,400 barrels`

### 22. [8-Scenario Parametric Control] `counterfactual_control` / `cf_real_cause_rig` (`REVISED`, $\Delta t=4$d)
- **Initial (2 repeats):** *Initial rig report blames main generator overload for deck blast* — Preliminary deck logs suggested a main generator overload sparked the initial explosion on the offshore platform.
- **Correction:** *Forensics rule out generator overload, cite thruster fuel-line fracture* — Official inquiry correction: Marine investigators retracted the generator overload theory and confirmed an auxiliary thruster fuel-line fracture caused the deck blast.
- **Evaluation Rule:** Target status `REVISED`, expected=`fuel-line fracture`, forbidden=`generator overload`

### 23. [8-Scenario Parametric Control] `counterfactual_control` / `cf_real_numeric_lehman` (`REVISED`, $\Delta t=5$d)
- **Initial (3 repeats):** *Initial treasury memo pegs emergency credit line at $18 billion* — Early overnight memos estimated the emergency clearinghouse credit line at $18 billion during the market liquidity crunch.
- **Correction:** *Final treasury ledger revises emergency credit line to $63 billion* — Correction and final ledger: Treasury auditors revised the emergency clearinghouse credit line upward to $63 billion, superseding the $18 billion memo.
- **Evaluation Rule:** Target status `REVISED`, expected=`$63 billion`, forbidden=`$18 billion`

### 24. [8-Scenario Parametric Control] `counterfactual_control` / `cf_real_retract_fourchon` (`RETRACTED`, $\Delta t=2$d)
- **Initial (3 repeats):** *Unconfirmed reports of Port Fourchon subsea storage tank explosion* — Local social channels claimed a subsea storage tank explosion occurred near Port Fourchon overnight.
- **Correction:** *Port authorities debunk Port Fourchon subsea tank explosion rumor* — Official retraction: Port Fourchon operators confirmed no subsea storage tank explosion occurred; earlier reports were false.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`subsea storage tank explosion`

### 25. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_num_reservoir` (`REVISED`, $\Delta t=18$d)
- **Initial (2 repeats):** *Initial survey puts reservoir spill volume at 12,000 cubic meters* — Regional water engineers estimated on the day of the breach that 12,000 cubic meters of industrial runoff entered the northern basin.
- **Correction:** *Final hydrological audit measures basin runoff at 34,000 cubic meters* — Hydrologists completing satellite bathymetry measurements concluded that the actual runoff volume from the northern basin breach totaled 34,000 cubic meters.
- **Evaluation Rule:** Target status `REVISED`, expected=`34,000 cubic meters`, forbidden=`12,000 cubic meters`

### 26. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_num_casualties` (`REVISED`, $\Delta t=25$d)
- **Initial (2 repeats):** *Port authorities count 14 injured workers after crane collapse* — Hospital intake logs on the afternoon of the shipyard crane failure listed 14 injured dockworkers receiving emergency treatment.
- **Correction:** *Labor ministry tally puts shipyard crane injuries at 29* — Consolidated clinic and subcontractor records released this week show 29 dockworkers sustained injuries during the shipyard crane collapse.
- **Evaluation Rule:** Target status `REVISED`, expected=`29 dockworkers`, forbidden=`14 injured`

### 27. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_causal_grid` (`REVISED`, $\Delta t=15$d)
- **Initial (2 repeats):** *Operators point to lightning strike for substation blackout* — Utility dispatchers attributed the regional substation outage to a severe lightning strike along the western transmission corridor.
- **Correction:** *Forensic teardown traces substation blackout to corroded relay switch* — Independent grid inspectors found no storm damage on the towers and determined that a corroded relay switch inside transformer bay three caused the substation blackout.
- **Evaluation Rule:** Target status `REVISED`, expected=`corroded relay switch`, forbidden=`lightning strike`

### 28. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_causal_cargo` (`REVISED`, $\Delta t=30$d)
- **Initial (2 repeats):** *Coast patrol suspects fuel line leak in freighter engine fire* — Early harbour briefings pointed toward a ruptured diesel fuel line as the ignition source aboard the cargo vessel MV Horizon.
- **Correction:** *Marine board links MV Horizon fire to lithium battery cargo* — Metallurgical testing of the lower hold showed intact fuel pipes and established that thermal runaway in undocumented lithium battery pallets ignited the MV Horizon fire.
- **Evaluation Rule:** Target status `REVISED`, expected=`lithium battery`, forbidden=`diesel fuel line`

### 29. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_date_quake` (`REVISED`, $\Delta t=21$d)
- **Initial (2 repeats):** *Seismic fault rupture logged on {initial_date} near coastal ridge* — Preliminary borehole readings placed the initial deep fault slip along the coastal ridge on {initial_date}.
- **Correction:** *Ocean-bottom sensors date coastal fault slip to {corrected_date}* — Synchronized ocean-bottom seismometer logs place the onset of the coastal ridge fault rupture on {corrected_date} rather than {initial_date}.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 30. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_date_treaty` (`REVISED`, $\Delta t=35$d)
- **Initial (2 repeats):** *Trade delegates sign tariff protocol on {initial_date}* — Press summaries from Geneva indicated that trade ministers finalized the tariff protocol signature on {initial_date}.
- **Correction:** *Archival registry records tariff protocol signing on {corrected_date}* — The official Geneva secretariat deposit shows that ministers signed the final tariff protocol on {corrected_date}, not {initial_date}.
- **Evaluation Rule:** Target status `REVISED`, expected=`{corrected_date}`, forbidden=`{initial_date}`

### 31. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_retract_bridge` (`RETRACTED`, $\Delta t=16$d)
- **Initial (2 repeats):** *Motorists report structural crack on East River suspension bridge* — Commuter calls to local radio described a visible steel girder crack on the East River suspension bridge following morning tremors.
- **Correction:** *Ultrasonic scan finds East River bridge girders completely intact* — State structural engineers completed a full ultrasonic inspection of the East River bridge and confirmed no girder cracks or structural damage exist; the roadway never fractured.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`girder crack`

### 32. [8-Scenario Implicit & Long-Delay] `implicit_long_delay` / `imp_retract_vaccine` (`RETRACTED`, $\Delta t=28$d)
- **Initial (2 repeats):** *Clinics halt batch K-19 over suspected cold-chain spoilage* — Regional clinics briefly quarantined vaccine batch K-19 amid fears that a warehouse freezer failure had spoiled the entire shipment.
- **Correction:** *Thermometer logs confirm batch K-19 remained within cold-chain limits* — Independent laboratory potency assays and continuous warehouse telemetry showed that batch K-19 never warmed above threshold and no spoilage occurred.
- **Evaluation Rule:** Target status `RETRACTED`, expected=``, forbidden=`cold-chain spoilage`
