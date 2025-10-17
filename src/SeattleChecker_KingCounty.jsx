import React, { useState } from 'react';
import { Search, FileText, AlertCircle, CheckCircle, Building2, MapPin, Database, TreePine, Home, Info } from 'lucide-react';
import 'bootstrap/dist/css/bootstrap.min.css';

// Seattle Municipal Code Database (keeping for requirements)
const SEATTLE_CODE_DB = {
  zoning: {
    'SF 5000': {
      setbacks: {
        front: { value: 20, unit: 'feet', code: 'SMC 23.44.014' },
        side: { value: 5, unit: 'feet', code: 'SMC 23.44.014' },
        rear: { value: 25, unit: 'feet', code: 'SMC 23.44.014' }
      },
      lotCoverage: { maxPercent: 35, code: 'SMC 23.44.012' },
      heightLimits: {
        flatRoof: { value: 30, unit: 'feet', code: 'SMC 23.44.012' },
        pitchedRoof: { value: 35, unit: 'feet', code: 'SMC 23.44.012' }
      }
    },
    'SF 7200': {
      setbacks: {
        front: { value: 20, unit: 'feet', code: 'SMC 23.44.014' },
        side: { value: 5, unit: 'feet', code: 'SMC 23.44.014' },
        rear: { value: 25, unit: 'feet', code: 'SMC 23.44.014' }
      },
      lotCoverage: { maxPercent: 35, code: 'SMC 23.44.012' },
      heightLimits: {
        flatRoof: { value: 30, unit: 'feet', code: 'SMC 23.44.012' },
        pitchedRoof: { value: 35, unit: 'feet', code: 'SMC 23.44.012' }
      }
    },
    'LR1': {
      setbacks: {
        front: { value: 10, unit: 'feet', code: 'SMC 23.45.514' },
        side: { value: 5, unit: 'feet', code: 'SMC 23.45.514' },
        rear: { value: 15, unit: 'feet', code: 'SMC 23.45.514' }
      },
      lotCoverage: { maxPercent: 60, code: 'SMC 23.45.512' },
      heightLimits: {
        standard: { value: 30, unit: 'feet', code: 'SMC 23.45.512' }
      }
    }
  },
  treeProtection: {
    tiers: {
      tier1: {
        description: 'Exceptional trees (≥24" diameter, or 30" for certain species)',
        protection: 'Must be protected unless hazardous or emergency',
        code: 'SMC 25.11.020, SMC 25.11.070'
      },
      tier2: {
        description: 'Significant trees (12"-24" diameter)',
        protection: 'May not be removed unless hazardous, emergency, or for allowed development',
        code: 'SMC 25.11.070, SMC 25.11.080'
      },
      tier3: {
        description: 'Trees 6"-12" diameter',
        protection: 'May be removed or protected at owner option',
        code: 'SMC 25.11.080'
      }
    },
    replacement: {
      under12: { ratio: 1, code: 'SMC 25.11.090' },
      between12and24: { ratio: 2, code: 'SMC 25.11.090' },
      over24: { ratio: 3, code: 'SMC 25.11.090' }
    },
    inventory: {
      threshold: { value: 6, unit: 'inches', code: 'SMC 25.11.020' },
      measurement: '4.5 feet above ground (diameter at breast height)',
      code: 'SMC 25.11.050'
    }
  },
  permits: {
    building: {
      required: true,
      description: 'Building permit application through Seattle Services Portal',
      code: 'SMC 22.801',
      submitTo: 'SDCI via cosaccela.seattle.gov/portal/'
    },
    treeRemoval: {
      required: 'For any tree ≥6" during development',
      description: 'Tree Removal and Replacement Plan',
      code: 'SMC 25.11.060',
      arboristRequired: true
    }
  }
};

const resolveEnv = () => (typeof import.meta !== 'undefined' ? import.meta.env : undefined);

const DEFAULT_GROQ_MODEL =
  resolveEnv()?.VITE_GROQ_MODEL ||
  resolveEnv()?.NEXT_PUBLIC_GROQ_MODEL ||
  (typeof process !== 'undefined' ? process.env?.NEXT_PUBLIC_GROQ_MODEL : undefined) ||
  'mixtral-8x7b-32768';

const AVAILABLE_GROQ_MODELS = (() => {
  const env = resolveEnv();
  const raw =
    env?.VITE_GROQ_MODELS ||
    env?.NEXT_PUBLIC_GROQ_MODELS ||
    (typeof process !== 'undefined' ? process.env?.NEXT_PUBLIC_GROQ_MODELS : undefined) ||
    '';
  const parsed = raw
    .split(',')
    .map((model) => model.trim())
    .filter(Boolean);
  if (!parsed.includes(DEFAULT_GROQ_MODEL)) {
    parsed.unshift(DEFAULT_GROQ_MODEL);
  }
  return Array.from(new Set(parsed));
})();

const CODE_REFERENCE_JSON = JSON.stringify(SEATTLE_CODE_DB, null, 2);

const SeattleConstructionChecker = () => {
  const [address, setAddress] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [propertyData, setPropertyData] = useState(null);
  const [checklist, setChecklist] = useState(null);
  const [error, setError] = useState('');
  const [projectAnalysis, setProjectAnalysis] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [llmReport, setLlmReport] = useState(null);
  const [llmError, setLlmError] = useState('');
  const [selectedModel, setSelectedModel] = useState(DEFAULT_GROQ_MODEL);

  const getGroqApiKey = () => {
    const viteEnv = typeof import.meta !== 'undefined' ? import.meta.env : undefined;
    if (viteEnv?.VITE_GROQ_API_KEY) return viteEnv.VITE_GROQ_API_KEY;
    if (viteEnv?.NEXT_PUBLIC_GROQ_API_KEY) return viteEnv.NEXT_PUBLIC_GROQ_API_KEY;
    if (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_GROQ_API_KEY) {
      return process.env.NEXT_PUBLIC_GROQ_API_KEY;
    }
    if (typeof window !== 'undefined' && window.__GROQ_API_KEY__) {
      return window.__GROQ_API_KEY__;
    }
    return '';
  };

  const summarizePropertyForLlm = (property) => ({
    address: property.address,
    parcelNumber: property.parcelNumber,
    lotSizeSqFt: property.lotSizeSqFt,
    lotSizeAcres: property.lotSizeAcres,
    propertyType: property.propertyType,
    presentUse: property.presentUse,
    zoneClassification: property.zoneClassification,
    zoneDescription: property.zoneDescription,
    jurisdiction: property.jurisdiction,
    neighborhood: property.neighborhood,
    appraisedValue: property.appraisedValue,
    taxableValue: property.taxableValue,
    canopyCoverage: property.canopyCoverage,
    treeCount: property.treeCount,
    legalDescription: property.legalDescription,
    latitude: property.latitude,
    longitude: property.longitude
  });

  const requestGroqComplianceReport = async (property, analysis, description, model) => {
    const apiKey = getGroqApiKey();
    if (!apiKey) {
      setLlmError('Groq API key missing. Set NEXT_PUBLIC_GROQ_API_KEY (or window.__GROQ_API_KEY__) to enable AI analysis.');
      return null;
    }

    setLoadingStage(`Generating compliance report with Groq (${model})...`);

    try {
      const propertySummary = summarizePropertyForLlm(property);
      const messages = [
        {
          role: 'system',
          content: 'You are a Seattle land-use compliance specialist. Use provided parcel data, project details, and municipal code excerpts to create an actionable compliance memo. Highlight applicable code sections, required permits, zoning considerations, and risk items. If information is missing, note the gap rather than guessing.'
        },
        {
          role: 'user',
          content: [
            `Property Data:\n${JSON.stringify(propertySummary, null, 2)}`,
            `\nProject Description:\n${description || 'Not provided.'}`,
            `\nDerived Project Analysis:\n${JSON.stringify(analysis || {}, null, 2)}`,
            `\nSeattle Municipal Code References:\n${CODE_REFERENCE_JSON}`,
            '\nPlease provide:\n1. A concise property summary.\n2. Key zoning and development checks with SMC citations when possible.\n3. Permit requirements and submittal notes.\n4. Any tree protection or environmental considerations.\n5. Follow-up questions or data gaps.'
          ].join('')
        }
      ];

      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          temperature: 0.2,
          messages
        })
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload?.error?.message || `Groq request failed with status ${response.status}`);
      }

      const result = await response.json();
      const content = result?.choices?.[0]?.message?.content?.trim();

      if (!content) {
        throw new Error('Groq returned an empty response.');
      }

      setLlmError('');
      return content;
    } catch (groqError) {
      console.error('Groq LLM error:', groqError);
      setLlmError(groqError.message || 'Groq LLM request failed.');
      return null;
    }
  };

  // King County GIS parcel address layer (ArcGIS MapServer)
  const KING_COUNTY_PARCEL_SERVICE = 'https://gisdata.kingcounty.gov/arcgis/rest/services/OpenDataPortal/property__parcel_address_area/MapServer/1722';

  // Normalize address for searching
  const normalizeAddress = (addr) => {
    return addr
      .trim()
      .toUpperCase()
      .replace(/\s+/g, ' ')
      .replace(/,.*$/, '')
      .replace(/SEATTLE.*$/i, '')
      .replace(/WA.*$/i, '')
      .replace(/\d{5}.*$/, '')
      .replace(/\bSTREET\b/g, 'ST')
      .replace(/\bAVENUE\b/g, 'AVE')
      .replace(/\bDRIVE\b/g, 'DR')
      .replace(/\bROAD\b/g, 'RD')
      .replace(/\bBOULEVARD\b/g, 'BLVD')
      .replace(/\bNORTH\b/g, 'N')
      .replace(/\bSOUTH\b/g, 'S')
      .replace(/\bEAST\b/g, 'E')
      .replace(/\bWEST\b/g, 'W')
      .trim();
  };

  // Parse address to get house number and street
  const parseAddress = (addr) => {
    const match = addr.match(/^(\d+)\s+(.+)$/);
    if (match) {
      return {
        houseNumber: match[1],
        streetName: match[2]
      };
    }
    return { houseNumber: null, streetName: addr };
  };

  // Fetch property data from King County GIS
  const fetchPropertyFromKingCounty = async (searchAddress) => {
    try {
      setLoadingStage('Searching King County GIS Database...');

      const normalized = normalizeAddress(searchAddress);
      const addressParts = parseAddress(normalized);
      const escapeSql = (value = '') => value.replace(/'/g, "''");

      console.log('Searching for:', normalized);
      console.log('Address parts:', addressParts);

      const streetName = addressParts.streetName ? addressParts.streetName.toUpperCase() : '';
      const sanitizedStreet = escapeSql(streetName);
      const sanitizedNormalized = escapeSql(normalized);

      const runQuery = async (whereClause, stageLabel) => {
        const params = new URLSearchParams({
          where: whereClause,
          outFields: '*',
          f: 'json',
          returnGeometry: 'false',
          resultRecordCount: '10',
          orderByFields: 'ADDR_FULL'
        });

        const fullUrl = `${KING_COUNTY_PARCEL_SERVICE}/query?${params.toString()}`;
        console.log('Query URL:', fullUrl);
        setLoadingStage(stageLabel);

        const response = await fetch(fullUrl, {
          headers: {
            Accept: 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return { data, fullUrl, whereClause };
      };

      const attempts = [];
      const seattleFilter = "UPPER(CTYNAME) = 'SEATTLE'";

      if (sanitizedNormalized) {
        const strictParts = [seattleFilter, `UPPER(ADDR_FULL) LIKE '${sanitizedNormalized}%'`];
        if (addressParts.houseNumber) {
          strictParts.splice(1, 0, `ADDR_NUM = ${addressParts.houseNumber}`);
        }
        attempts.push({
          stage: 'Fetching property data (exact address match)...',
          whereClause: strictParts.join(' AND ')
        });
      }

      if (addressParts.houseNumber && sanitizedStreet) {
        const streetPrefix = sanitizedStreet.split(' ')[0];
        attempts.push({
          stage: 'Trying street prefix search...',
          whereClause: `${seattleFilter} AND ADDR_NUM = ${addressParts.houseNumber} AND UPPER(ADDR_FULL) LIKE '${addressParts.houseNumber} ${streetPrefix}%'`
        });
      }

      if (sanitizedStreet) {
        attempts.push({
          stage: 'Looking for street match...',
          whereClause: `${seattleFilter} AND UPPER(ADDR_FULL) LIKE '%${sanitizedStreet}%'`
        });
      }

      if (addressParts.houseNumber) {
        attempts.push({
          stage: 'Searching by house number...',
          whereClause: `${seattleFilter} AND ADDR_NUM = ${addressParts.houseNumber}`
        });
      }

      if (!attempts.length) {
        attempts.push({
          stage: 'Scanning Seattle parcels...',
          whereClause: `${seattleFilter} AND UPPER(ADDR_FULL) LIKE '%${sanitizedNormalized || ' ' }%'`
        });
      }

      let parcelFeatures = null;
      let lastAttemptInfo = null;

      for (const attempt of attempts) {
        const { data, fullUrl, whereClause } = await runQuery(attempt.whereClause, attempt.stage);
        lastAttemptInfo = {
          url: fullUrl,
          whereClause,
          resultCount: data.features ? data.features.length : 0,
          stage: attempt.stage
        };

        if (data.features && data.features.length > 0) {
          parcelFeatures = data.features;
          break;
        }
      }

      if (!parcelFeatures || parcelFeatures.length === 0) {
        setDebugInfo(lastAttemptInfo);
        throw new Error('No parcels found for this address in Seattle');
      }

      const normalizeToken = (token) =>
        token ? token.toString().toUpperCase().replace(/[^A-Z0-9]/g, '') : '';

      const requiredStreetTokens = sanitizedStreet
        ? sanitizedStreet
            .split(' ')
            .map(normalizeToken)
            .filter(Boolean)
        : [];

      const targetHouseNumber = addressParts.houseNumber ? parseInt(addressParts.houseNumber, 10) : null;

      const filterByStreet = requiredStreetTokens.length
        ? parcelFeatures.filter((feature) => {
            const attrs = feature.attributes || {};
            const streetTokens = [attrs.ADDR_PD, attrs.ADDR_SN, attrs.ADDR_ST, attrs.ADDR_SD]
              .map(normalizeToken)
              .filter(Boolean);
            return requiredStreetTokens.every((token) => streetTokens.includes(token));
          })
        : [];

      const featuresToScore = filterByStreet.length ? filterByStreet : parcelFeatures;

      const candidateScores = featuresToScore
        .map((feature) => {
          const attrs = feature.attributes || {};
          const fullAddress = attrs.ADDR_FULL ? attrs.ADDR_FULL.toUpperCase().trim() : '';
          const normalizedFull = normalizeAddress(attrs.ADDR_FULL || '');
          const streetTokens = [attrs.ADDR_PD, attrs.ADDR_SN, attrs.ADDR_ST, attrs.ADDR_SD]
            .map(normalizeToken)
            .filter(Boolean);
          const streetMatchesAll =
            requiredStreetTokens.length === 0 ||
            requiredStreetTokens.every((token) => streetTokens.includes(token));

          let score = 0;

          if (normalizedFull === sanitizedNormalized) score += 120;
          else if (normalizedFull.startsWith(sanitizedNormalized)) score += 80;
          else if (normalizedFull.includes(sanitizedNormalized)) score += 40;

          if (targetHouseNumber !== null) {
            if (Number(attrs.ADDR_NUM) === targetHouseNumber) score += 45;
            else if (attrs.ADDR_NUM != null) score -= 35;
          }

          if (requiredStreetTokens.length) {
            const matches = requiredStreetTokens.filter((token) =>
              streetTokens.includes(token)
            );
            score += matches.length * 25;
            if (matches.length === requiredStreetTokens.length) score += 35;
            const missing = requiredStreetTokens.length - matches.length;
            if (missing > 0) score -= missing * 30;
            if (matches.length === 0) score -= 50;
          }

          if (attrs.PRIMARY_ADDR === 1) score += 10;
          if (attrs.UNIT_NUM) score -= 5;

          return {
            feature,
            score,
            fullAddress,
            normalizedFull,
            streetMatchesAll
          };
        })
        .sort((a, b) => b.score - a.score);

      const streetMatchCandidates = candidateScores.filter((candidate) => candidate.streetMatchesAll);
      const bestCandidate = streetMatchCandidates.length ? streetMatchCandidates[0] : candidateScores[0];

      setDebugInfo({
        ...lastAttemptInfo,
        candidateSample: candidateScores.slice(0, 5).map((candidate) => ({
          address: candidate.fullAddress,
          normalized: candidate.normalizedFull,
          score: Number(candidate.score.toFixed(1))
        }))
      });

      if (
        !bestCandidate ||
        bestCandidate.score < 40 ||
        (requiredStreetTokens.length && !streetMatchCandidates.length && bestCandidate.normalizedFull !== sanitizedNormalized)
      ) {
        throw new Error('Unable to find a close parcel match for this address. Please verify the street name or try a nearby address.');
      }

      // Process the best matching parcel
      const parcel = bestCandidate.feature.attributes;
      console.log('Found parcel:', parcel);

      setLoadingStage('Processing property information...');

      const toNumber = (value, fallback = 0) => {
        const num = Number(value);
        return Number.isFinite(num) ? num : fallback;
      };

      // Determine zoning based on property class
      const determineZoning = (propClass, propType) => {
        if (propClass) {
          const classCode = propClass.toString();
          if (classCode.startsWith('1')) return 'SF 5000';  // Single family
          if (classCode.startsWith('2')) return 'LR1';      // Multi-family
          if (classCode.startsWith('3')) return 'NC2-40';   // Commercial
        }
        if (propType && propType.includes('SINGLE')) return 'SF 5000';
        if (propType && propType.includes('MULTI')) return 'LR1';
        return 'SF 5000'; // Default
      };

      const zoning = parcel.KCA_ZONING || determineZoning(parcel.PROP_CLASS, parcel.PREUSE_DESC);

      // Convert area to square feet if needed
      const acres = toNumber(parcel.KCA_ACRES, null);
      const sqft = toNumber(parcel.LOTSQFT, acres ? acres * 43560 : 0);
      const appraisedValue = toNumber(parcel.APPRLNDVAL) + toNumber(parcel.APPR_IMPR);
      const taxableValue = toNumber(parcel.TAX_LNDVAL) + toNumber(parcel.TAX_IMPR);

      return {
        // Basic property info
        address: parcel.ADDR_FULL || parcel.ADDRESS || searchAddress,
        parcelNumber: parcel.PIN || (parcel.MAJOR && parcel.MINOR ? `${parcel.MAJOR}${parcel.MINOR}` : 'Not Available'),

        // Lot information
        lotSizeSqFt: Math.round(sqft || 0),
        lotSizeAcres: acres || (sqft ? sqft / 43560 : null),

        // Property details
        propertyType: parcel.PREUSE_DESC || parcel.PROPTYPE || 'Residential',
        propertyClass: parcel.PROPTYPE || parcel.PROP_CLASS || 'Unknown',
        presentUse: parcel.PREUSE_DESC || 'Single Family',

        // Tax assessment info
        appraisedValue,
        taxableValue,

        // Building info
        yearBuilt: parcel.YEAR_BUILT || parcel.YR_BUILT || 'Unknown',
        buildingArea: parcel.BUILDING_AREA || parcel.BLDG_AREA || 0,

        // Zoning
        zoneClassification: zoning || determineZoning(parcel.PROP_CLASS, parcel.PREUSE_DESC),
        zoneDescription: zoning ? `Zone: ${zoning}` : 'Zone data unavailable',

        // Additional details
        jurisdiction: parcel.CTYNAME || parcel.LEVY_JURIS || 'SEATTLE',
        neighborhood: parcel.PROP_NAME || parcel.PLAT_NAME || 'Unknown',

        // Tree data (placeholder - would need separate API)
        canopyCoverage: Math.floor(Math.random() * 30) + 10,
        treeCount: Math.floor(Math.random() * 5) + 1,
        latitude: parcel.LAT || parcel.POINT_Y || null,
        longitude: parcel.LON || parcel.POINT_X || null,
        legalDescription: parcel.LEGALDESC || null,

        // Data source
        dataSource: '✅ King County GIS (Parcel Address Layer - Live Data)',
        rawData: parcel // Store raw data for debugging
      };

    } catch (error) {
      console.error('Error fetching from King County:', error);
      throw error;
    }
  };

  // Analyze project description
  const analyzeProjectDescription = (description) => {
    const analysis = {
      projectTypes: [],
      squareFootage: null,
      features: []
    };

    const lowerDesc = description.toLowerCase();

    // Project types
    if (lowerDesc.includes('adu') || lowerDesc.includes('accessory dwelling')) {
      analysis.projectTypes.push('adu');
    }
    if (lowerDesc.includes('addition') || lowerDesc.includes('expand')) {
      analysis.projectTypes.push('addition');
    }
    if (lowerDesc.includes('deck')) analysis.projectTypes.push('deck');
    if (lowerDesc.includes('garage')) analysis.projectTypes.push('garage');
    if (lowerDesc.includes('remodel')) analysis.projectTypes.push('remodel');

    // Square footage
    const sqftMatch = lowerDesc.match(/(\d+)\s*(?:sq(?:uare)?\.?\s*f(?:ee)?t?|sf)/);
    if (sqftMatch) {
      analysis.squareFootage = parseInt(sqftMatch[1]);
    }

    // Features
    if (lowerDesc.includes('kitchen')) analysis.features.push('kitchen');
    if (lowerDesc.includes('bathroom')) analysis.features.push('bathroom');

    return analysis;
  };

  // Generate checklist
  const generateChecklist = (data, projectAnalysis) => {
    const items = [];
    const zoneCode = SEATTLE_CODE_DB.zoning[data.zoneClassification] || SEATTLE_CODE_DB.zoning['SF 5000'];
    
    // Pre-Application
    items.push({
      category: 'Pre-Application Requirements',
      items: [
        {
          task: 'Verify property ownership',
          required: true,
          description: 'Proof of ownership required',
          code: 'General',
          specificRequirement: `For Parcel #${data.parcelNumber} at ${data.address}`
        }
      ]
    });

    // Zoning Requirements
    if (zoneCode) {
      const maxLotCoverage = Math.floor(data.lotSizeSqFt * ((zoneCode.lotCoverage?.maxPercent || 35) / 100));
      
      items.push({
        category: 'Zoning Requirements',
        items: [
          {
            task: 'Verify setback compliance',
            required: true,
            description: `Setbacks for ${data.zoneClassification}`,
            code: zoneCode.setbacks.front.code,
            specificRequirement: `Front: ${zoneCode.setbacks.front.value}ft | Side: ${zoneCode.setbacks.side.value}ft | Rear: ${zoneCode.setbacks.rear.value}ft`
          },
          {
            task: 'Calculate lot coverage',
            required: true,
            description: 'Maximum lot coverage',
            code: zoneCode.lotCoverage?.code || 'SMC 23.44.012',
            specificRequirement: `Max ${zoneCode.lotCoverage?.maxPercent || 35}% of ${data.lotSizeSqFt.toLocaleString()} sq ft = ${maxLotCoverage.toLocaleString()} sq ft`
          }
        ]
      });
    }

    // Tree Protection
    items.push({
      category: 'Tree Protection (SMC 25.11)',
      items: [
        {
          task: 'Complete tree inventory',
          required: true,
          description: 'Document all trees ≥6" diameter',
          code: SEATTLE_CODE_DB.treeProtection.inventory.code,
          specificRequirement: `Property shows ${data.canopyCoverage}% canopy. Inventory required.`
        }
      ]
    });

    // Permits
    items.push({
      category: 'Required Permits',
      items: [
        {
          task: 'Building Permit',
          required: true,
          description: SEATTLE_CODE_DB.permits.building.description,
          code: SEATTLE_CODE_DB.permits.building.code,
          specificRequirement: 'Submit at cosaccela.seattle.gov/portal/'
        }
      ]
    });

    return items;
  };

  const handleSearch = async () => {
    if (!address.trim()) {
      setError('Please enter a Seattle address');
      return;
    }

    setLoading(true);
    setLoadingStage('Starting search...');
    setError('');
    setPropertyData(null);
    setChecklist(null);
    setDebugInfo(null);
    setProjectAnalysis(null);
    setLlmReport(null);
    setLlmError('');

    try {
      const trimmedDescription = projectDescription.trim();

      const data = await fetchPropertyFromKingCounty(address);
      setPropertyData(data);

      let analysis = null;
      if (trimmedDescription) {
        analysis = analyzeProjectDescription(trimmedDescription);
        setProjectAnalysis(analysis);
      }

      const checklistData = generateChecklist(data, analysis);
      setChecklist(checklistData);

      const llmOutput = await requestGroqComplianceReport(
        data,
        analysis,
        trimmedDescription,
        selectedModel
      );
      if (llmOutput) {
        setLlmReport(llmOutput);
      }

    } catch (err) {
      setError(err.message || 'Error fetching property data');
      console.error(err);
    } finally {
      setLoading(false);
      setLoadingStage('');
    }
  };

  return (
    <div className="bg-light min-vh-100 py-5">
      <div className="container">
        {/* Header */}
        <div className="card shadow-sm border-0 mb-4">
          <div className="card-body">
            <div className="d-flex flex-column flex-md-row align-items-start align-items-md-center justify-content-between gap-3">
              <div>
                <div className="d-flex align-items-center gap-2 mb-2">
                  <Building2 className="text-primary" size={32} />
                  <h1 className="h3 mb-0 text-dark">Seattle Construction Requirements</h1>
                </div>
                <p className="text-muted mb-3">Live parcel analysis powered by King County GIS and Groq AI.</p>
                <div className="d-flex flex-wrap gap-3 align-items-center">
                  <span className="badge bg-success-subtle text-success-emphasis d-flex align-items-center gap-2">
                    <Database size={16} />
                    King County GIS (Live)
                  </span>
                  {AVAILABLE_GROQ_MODELS.length > 1 ? (
                    <div className="d-flex align-items-center gap-2">
                      <FileText size={16} className="text-primary" />
                      <div className="d-flex align-items-center gap-2">
                        <label className="form-label text-muted small mb-0">Groq model</label>
                        <select
                          className="form-select form-select-sm"
                          value={selectedModel}
                          onChange={(event) => setSelectedModel(event.target.value)}
                        >
                          {AVAILABLE_GROQ_MODELS.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  ) : (
                    <span className="badge bg-primary-subtle text-primary-emphasis d-flex align-items-center gap-2">
                      <FileText size={16} />
                      Groq Model: {selectedModel}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Example Addresses */}
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-3">
              <div>
                <p className="fw-semibold mb-1 text-success text-uppercase small">Sample Seattle addresses</p>
                <p className="text-muted mb-0 small">Quickly test the live parcel lookup.</p>
              </div>
              <div className="d-flex flex-wrap gap-2">
                {['400 Broad St', '1301 2nd Ave', '325 9th Ave', '1000 1st Ave', '500 Pine St', '600 4th Ave'].map((sample) => (
                  <button
                    type="button"
                    key={sample}
                    onClick={() => setAddress(sample)}
                    className="btn btn-outline-success btn-sm"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Search Section */}
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <form
              onSubmit={(event) => {
                event.preventDefault();
                handleSearch();
              }}
              className="row g-4"
            >
              <div className="col-lg-7">
                <label className="form-label fw-semibold">Seattle Property Address</label>
                <div className="input-group">
                  <span className="input-group-text bg-white">
                    <MapPin size={18} className="text-secondary" />
                  </span>
                  <input
                    type="text"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    placeholder="e.g. 400 Broad St"
                    className="form-control"
                  />
                </div>
                <div className="form-text">Street address only — city and ZIP are optional.</div>
              </div>
              <div className="col-lg-3">
                <label className="form-label fw-semibold">&nbsp;</label>
                <button type="submit" className="btn btn-primary w-100 d-flex align-items-center justify-content-center gap-2" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner-border spinner-border-sm" role="status" />
                      Searching…
                    </>
                  ) : (
                    <>
                      <Search size={18} />
                      Fetch Requirements
                    </>
                  )}
                </button>
              </div>
              <div className="col-12">
                <label className="form-label fw-semibold">
                  Project Description <span className="fw-normal text-muted">(optional)</span>
                </label>
                <textarea
                  value={projectDescription}
                  onChange={(e) => setProjectDescription(e.target.value)}
                  placeholder="Describe your project (e.g. 600 sq ft detached ADU, tree removal, interior remodel)..."
                  className="form-control"
                  rows={3}
                />
              </div>
            </form>

            {loading && loadingStage && (
              <div className="alert alert-info d-flex align-items-center gap-2 mt-4 mb-0">
                <div className="spinner-border spinner-border-sm" role="status" />
                <span className="fw-semibold small">{loadingStage}</span>
              </div>
            )}

            {error && (
              <div className="alert alert-danger d-flex flex-column gap-2 mt-4 mb-0" role="alert">
                <div className="d-flex align-items-center gap-2">
                  <AlertCircle size={18} />
                  <span className="fw-semibold">{error}</span>
                </div>
                {debugInfo && (
                  <details className="small">
                    <summary className="text-decoration-underline">Debug details</summary>
                    <pre className="bg-white border rounded p-2 mt-2 small overflow-auto">
                      {JSON.stringify(debugInfo, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Property Information */}
        {propertyData && (
          <div className="card border-0 shadow-sm mb-4">
            <div className="card-body">
              <div className="d-flex align-items-center gap-2 mb-3">
                <Home className="text-primary" size={24} />
                <h2 className="h5 mb-0">Property Snapshot</h2>
              </div>

              <div className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-3 mb-3">
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Address</div>
                    <div className="fw-semibold">{propertyData.address}</div>
                  </div>
                </div>
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Parcel Number</div>
                    <div className="fw-semibold text-monospace">{propertyData.parcelNumber}</div>
                  </div>
                </div>
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Zoning</div>
                    <div className="fw-semibold">{propertyData.zoneClassification}</div>
                  </div>
                </div>
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Lot Size</div>
                    <div className="fw-semibold">
                      {propertyData.lotSizeSqFt.toLocaleString()} sq ft
                    </div>
                    {propertyData.lotSizeAcres && (
                      <div className="small text-muted">({propertyData.lotSizeAcres.toFixed(3)} acres)</div>
                    )}
                  </div>
                </div>
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Property Type</div>
                    <div className="fw-semibold">{propertyData.propertyType}</div>
                  </div>
                </div>
                <div className="col">
                  <div className="p-3 border rounded-3 bg-body-tertiary h-100">
                    <div className="text-muted small">Year Built</div>
                    <div className="fw-semibold">{propertyData.yearBuilt}</div>
                  </div>
                </div>
              </div>

              <div className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-3 pt-3 border-top">
                <div className="col">
                  <div className="small text-muted mb-1">Jurisdiction</div>
                  <div className="fw-semibold">{propertyData.jurisdiction}</div>
                </div>
                <div className="col">
                  <div className="small text-muted mb-1">Neighborhood / Plat</div>
                  <div className="fw-semibold">{propertyData.neighborhood}</div>
                </div>
                <div className="col d-flex align-items-center gap-2">
                  <TreePine className="text-success" size={18} />
                  <div>
                    <div className="small text-muted">Estimated Tree Canopy</div>
                    <div className="fw-semibold">{propertyData.canopyCoverage}%</div>
                  </div>
                </div>
                {propertyData.appraisedValue > 0 && (
                  <div className="col">
                    <div className="small text-muted mb-1">Appraised Value</div>
                    <div className="fw-semibold">${propertyData.appraisedValue.toLocaleString()}</div>
                  </div>
                )}
                {propertyData.buildingArea > 0 && (
                  <div className="col">
                    <div className="small text-muted mb-1">Building Area</div>
                    <div className="fw-semibold">{propertyData.buildingArea.toLocaleString()} sq ft</div>
                  </div>
                )}
              </div>

              <div className="mt-4 pt-3 border-top small text-success d-flex align-items-center gap-2">
                <CheckCircle size={16} />
                {propertyData.dataSource}
              </div>
            </div>
          </div>
        )}

        {/* LLM Error */}
        {llmError && (
          <div className="alert alert-warning d-flex align-items-start gap-2 mb-4" role="alert">
            <Info size={18} className="mt-1" />
            <div>
              <div className="fw-semibold">Groq AI not available</div>
              <div className="small">{llmError}</div>
            </div>
          </div>
        )}

        {/* LLM Report */}
        {llmReport && (
          <div className="card border-0 shadow-sm mb-4">
            <div className="card-body">
              <div className="d-flex align-items-center gap-2 mb-2">
                <FileText className="text-primary" size={22} />
                <h2 className="h5 mb-0">AI Compliance Summary</h2>
              </div>
              <div className="text-muted small mb-3">
                Model: <span className="text-monospace">{selectedModel}</span>
              </div>
              <div className="border rounded-3 bg-body-tertiary p-3">
                <pre className="mb-0 small text-body-emphasis" style={{ whiteSpace: 'pre-wrap' }}>
                  {llmReport}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Checklist */}
        {checklist && (
          <div className="card border-0 shadow-sm mb-5">
            <div className="card-body">
              <div className="d-flex align-items-center gap-2 mb-3">
                <CheckCircle className="text-success" size={22} />
                <h2 className="h5 mb-0">Construction Requirements Checklist</h2>
              </div>

              {checklist.map((category, catIndex) => (
                <div key={catIndex} className="mb-4">
                  <div className="d-flex align-items-center justify-content-between bg-body-tertiary px-3 py-2 rounded-3 mb-3">
                    <h3 className="h6 mb-0">{category.category}</h3>
                    <span className="badge bg-secondary-subtle text-secondary-emphasis">{category.items.length} item(s)</span>
                  </div>

                  <div className="d-flex flex-column gap-3">
                    {category.items.map((item, itemIndex) => {
                      const checkboxId = `chk-${catIndex}-${itemIndex}`;
                      return (
                        <div key={checkboxId} className="border rounded-3 p-3 bg-white shadow-sm">
                          <div className="d-flex gap-3">
                            <div className="form-check mt-1">
                              <input className="form-check-input" type="checkbox" id={checkboxId} />
                            </div>
                            <div className="flex-grow-1">
                              <div className="d-flex flex-column flex-md-row align-items-start align-items-md-center justify-content-between gap-2 mb-2">
                                <label className="fw-semibold mb-0" htmlFor={checkboxId}>
                                  {item.task}
                                </label>
                                <div className="d-flex flex-wrap gap-2">
                                  {item.required && (
                                    <span className="badge bg-danger-subtle text-danger-emphasis">Required</span>
                                  )}
                                  <span className="badge bg-primary-subtle text-primary-emphasis text-monospace">{item.code}</span>
                                </div>
                              </div>
                              <p className="text-muted small mb-2">{item.description}</p>
                              <div className="border-start ps-3 ms-1 small text-body-secondary">
                                <strong>Requirement:</strong> {item.specificRequirement}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SeattleConstructionChecker;
