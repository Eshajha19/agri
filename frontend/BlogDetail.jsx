import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { FaArrowLeft, FaClock, FaUser, FaCalendarAlt, FaLeaf, FaCloudSun, FaLandmark, FaBug, FaTint, FaSeedling, FaBookmark, FaRegBookmark } from "react-icons/fa";
import "./Blog.css";
import { getBookmarks, toggleBookmark } from "./utils/bookmarkStorage";

const BLOG_POSTS = [
  {
    id: 1,
    title: "Modern Drip Irrigation Techniques for Small Farms",
    description: "Discover how drip irrigation can reduce water consumption by up to 60% while boosting crop yields. Learn the setup process, maintenance tips, and which crops benefit most.",
    category: "Irrigation",
    author: "Dr. Anita Sharma",
    date: "April 28, 2026",
    readTime: "6 min read",
    thumbnail: "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=900&q=80",
    content: `Drip irrigation is one of the most water-efficient methods of irrigation available to small and medium farmers today. Unlike flood or sprinkler irrigation, drip systems deliver water directly to the root zone of each plant, drastically reducing evaporation and runoff.

**Why Drip Irrigation?**

Water scarcity is a growing concern across India, especially in rain-shadow regions of Maharashtra, Rajasthan, and Andhra Pradesh. Drip irrigation can cut water usage by 40 to 60 percent compared to conventional furrow irrigation, while simultaneously improving yields by maintaining optimal soil moisture levels.

**Key Components of a Drip System**

A basic drip system consists of a water source, a filter unit, a fertigation tank, mainline pipes, sub-mainlines, laterals, and drip emitters. Choosing the right emitter flow rate for your soil type is critical. Sandy soils need higher-flow emitters spaced closer together, while clay soils benefit from lower flow rates to avoid waterlogging.

**Crops Best Suited for Drip Irrigation**

Vegetables such as tomato, chilli, capsicum, cucumber, and okra respond excellently to drip irrigation. Orchard crops like pomegranate, banana, mango, and grapes also show significant yield improvements. Sugarcane grown under drip irrigation has shown yield increases of 20 to 30 percent in trials conducted by the Vasantdada Sugar Institute.

**Subsidy Availability**

Under the Pradhan Mantri Krishi Sinchai Yojana (PMKSY) - Per Drop More Crop component, eligible farmers can receive 45 to 55 percent subsidy on drip system installation costs. Small and marginal farmers receive higher subsidy percentages. Contact your district agriculture office or nearest Krishi Vigyan Kendra to begin the application process.

**Maintenance Tips**

Flush the laterals at least once a month to remove sediment buildup. Clean or replace filters every season. Check for clogged emitters regularly by observing wilting plants and uneven growth. During winter, drain the system if temperatures drop below 5 degrees Celsius to prevent pipe cracking.`,
  },
  {
    id: 2,
    title: "PM-KISAN Scheme: How to Apply and Maximise Your Benefits",
    description: "A complete guide to the Pradhan Mantri Kisan Samman Nidhi scheme. Learn the eligibility criteria, application process, and common mistakes to avoid when claiming support.",
    category: "Government Schemes",
    author: "Rajesh Verma",
    date: "April 22, 2026",
    readTime: "8 min read",
    thumbnail: "https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?w=900&q=80",
    content: `The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) scheme provides direct income support of Rs 6,000 per year to eligible farmer families across India. The amount is transferred in three equal instalments of Rs 2,000 directly into the bank accounts of beneficiaries.

**Who is Eligible?**

All landholding farmer families are eligible, subject to certain exclusions. Institutional landholders, farmer families holding constitutional posts, serving or retired officers of state or central government, professionals like doctors, engineers, and lawyers, and individuals with an income tax liability are excluded from the scheme.

**How to Register**

Farmers can register through the PM-KISAN portal at pmkisan.gov.in, through Common Service Centres (CSCs), or through state nodal officers. You will need your Aadhaar number, bank account details linked to Aadhaar, and land records (Khasra/Khatauni) to complete registration.

**Common Mistakes to Avoid**

Incorrect Aadhaar seeding with bank accounts is the most frequent reason for payment failures. Ensure your mobile number is linked to Aadhaar before applying. Land records must be in your name or jointly in your name. If your name differs slightly between records and Aadhaar, approach the revenue office for correction before applying.

**Checking Your Status**

Log in to pmkisan.gov.in and click on Beneficiary Status. Enter your Aadhaar number, account number, or mobile number to check your payment history and any pending issues flagged by the system.

**Maximising Your Benefit**

Use PM-KISAN instalments strategically by timing withdrawals with your crop input procurement needs. The first instalment (April-July) aligns well with kharif season sowing expenses. The second (August-November) supports kharif harvesting and rabi sowing. The third (December-March) covers rabi season inputs.`,
  },
  {
    id: 3,
    title: "Identifying and Managing Rice Blast Disease",
    description: "Rice blast is one of the most destructive fungal diseases affecting paddy crops. This guide covers early identification signs, preventive cultural practices, and effective fungicide timings.",
    category: "Crop Management",
    author: "Prof. Suresh Patel",
    date: "April 18, 2026",
    readTime: "7 min read",
    thumbnail: "https://images.unsplash.com/photo-1595841696677-6489ff3f8cd1?w=900&q=80",
    content: `Rice blast, caused by the fungus Magnaporthe oryzae, is the most economically destructive disease of rice worldwide. Under favourable conditions it can cause total crop failure. Early identification and timely management are essential for protecting your paddy crop.

**Identifying Rice Blast**

Leaf blast appears as diamond or spindle-shaped lesions with grey or white centres and brown or reddish-brown borders. Neck blast is the most damaging form - the fungus attacks the panicle neck just below the grain-bearing portion, causing the neck to break and the panicle to fall, resulting in unfilled grains or complete spikelet sterility.

**Conditions Favouring Disease Development**

Rice blast thrives in cool temperatures between 18 and 28 degrees Celsius, high relative humidity above 90 percent, and extended leaf wetness periods of 10 hours or more. Heavy nitrogen application, especially in split doses, aggravates disease severity. Upland and aerobic rice systems face higher risk than flooded systems.

**Preventive Cultural Practices**

Select resistant varieties available from your state agriculture university. Avoid excess nitrogen application. Split nitrogen into at least three doses and avoid applying excessive nitrogen at panicle initiation. Maintain proper spacing to improve air circulation. Remove and destroy infected plant debris after harvest.

**Chemical Management**

At the first sign of leaf blast, apply Tricyclazole 75 WP at 0.6 grams per litre of water, or Isoprothiolane 40 EC at 1.5 ml per litre. For neck blast prevention, apply fungicide at boot leaf stage (5 to 7 days before panicle emergence) and repeat after 10 days. Carbendazim 50 WP at 1 gram per litre is also effective.

**Resistance Management**

Rotate fungicides with different modes of action to delay resistance development. Avoid using the same active ingredient consecutively in the same season.`,
  },
  {
    id: 4,
    title: "Understanding the Southwest Monsoon and Your Kharif Season",
    description: "Accurate monsoon prediction is critical for kharif crop planning. Learn how to use IMD forecasts, interpret weather data, and plan sowing windows to minimise risk.",
    category: "Weather",
    author: "Meena Krishnan",
    date: "April 14, 2026",
    readTime: "5 min read",
    thumbnail: "https://images.unsplash.com/photo-1561470508-fd4df1ed90b2?w=900&q=80",
    content: `The Southwest Monsoon is the lifeblood of Indian agriculture, delivering approximately 70 to 90 percent of annual rainfall to most parts of the country between June and September. Understanding how to read monsoon forecasts and plan your kharif season accordingly can significantly improve your outcomes.

**Reading IMD Forecasts**

The India Meteorological Department releases long-range forecasts for the season in April and May, and updates in June. District and sub-divisional forecasts are available weekly through the Meghdoot portal and the Kisan mobile application. Learn to distinguish between cumulative seasonal forecasts and weekly short-range forecasts, as they serve different planning purposes.

**Onset and Withdrawal Dates**

The monsoon typically arrives over Kerala around June 1, though this varies by one to two weeks. It progresses northward, reaching Delhi by late June or early July under normal conditions. Check the IMD monsoon onset line map regularly during May and June to refine your sowing plans.

**Planning Sowing Windows**

For most kharif crops, sowing within the optimal window after monsoon onset is critical. For paddy, sowing or transplanting within two weeks of onset ensures adequate rainfall for the entire growing period. Maize and sorghum have more flexible windows but perform best when sown by the first week of July in north India.

**Dealing with Dry Spells**

Intraseasonal dry spells lasting 10 to 15 days are common even in good monsoon years. Keep contingency crop varieties on hand. If a dry spell extends beyond 15 days during vegetative stages, consider life-saving irrigation if water is available. Mulching with crop residue can significantly reduce soil moisture loss.

**El Nino and La Nina Effects**

El Nino years are often associated with below-normal monsoon rainfall, while La Nina years tend to bring above-normal rainfall. Monitor NOAA and IMD ENSO outlooks from February onwards to adjust your crop planning and insurance decisions.`,
  },
  {
    id: 5,
    title: "Soil Health Card Scheme: Getting the Most from Your Soil Test",
    description: "Your soil health card contains crucial data about nutrients, pH, and micro-nutrients. This article explains how to read each parameter and apply inputs efficiently.",
    category: "Government Schemes",
    author: "Dr. Kavita Rao",
    date: "April 10, 2026",
    readTime: "6 min read",
    thumbnail: "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=900&q=80",
    content: `The Soil Health Card Scheme was launched to provide every farmer in India with a soil health card once every two years. The card contains information on 12 parameters including macro-nutrients (N, P, K), secondary nutrients (S, Ca, Mg), and micro-nutrients (Zn, Fe, Cu, Mn, B) along with pH and electrical conductivity.

**Understanding Soil pH**

pH between 6.5 and 7.5 is considered optimal for most crops. Acidic soils (pH below 6) require lime application to raise pH and improve nutrient availability. Alkaline soils (pH above 8) may need gypsum or elemental sulphur to lower pH. Your card will indicate whether corrective action is needed.

**Interpreting NPK Status**

Nitrogen status is indicated as low, medium, or sufficient. Most Indian soils show low to medium nitrogen, making basal and top-dress nitrogen applications essential for most crops. Phosphorus deficiency is widespread in acidic soils of eastern India. Potassium deficiency is increasingly common in intensive farming areas where crop residues are burned rather than incorporated.

**Micro-nutrient Deficiencies**

Zinc deficiency is the most widespread micro-nutrient problem in India, affecting over 50 percent of soils tested. If your card shows zinc as deficient, apply zinc sulphate at 25 kilograms per hectare as a basal dose or 5 kilograms per hectare as a foliar spray. Boron deficiency affects oilseeds, pulses, and vegetables in many regions.

**Making a Fertiliser Plan**

Use the recommendations on your card as a starting point, then adjust for your crop, expected yield, and actual soil observations. Input recommendations on the card are provided for common crops in your district. For speciality crops or high-value horticulture, consult your Krishi Vigyan Kendra for crop-specific advice.

**Getting Your Card**

Visit your nearest Common Service Centre or primary agriculture cooperative society. Soil testing is free. Bring a representative soil sample collected from 10 to 15 spots in your field at a depth of 15 to 20 centimetres, mixed thoroughly.`,
  },
  {
    id: 6,
    title: "Integrated Pest Management for Cotton Crops",
    description: "Chemical-only pest control is becoming less effective. Explore IPM strategies for cotton including scouting protocols, biological controls, and pheromone traps to cut input costs.",
    category: "Pest Management",
    author: "Arvind Kulkarni",
    date: "April 5, 2026",
    readTime: "9 min read",
    thumbnail: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=900&q=80",
    content: `Integrated Pest Management (IPM) is a science-based approach that combines multiple pest control strategies to manage pest populations below economically damaging levels while minimising risks to human health and the environment.

**Key Pests of Cotton**

The major pests of cotton in India include bollworms (American, Spotted, Pink), sucking pests (Aphids, Jassids, Thrips, Whitefly, Mealybug), and soil pests (White grub, Cut worm). Bt cotton hybrids provide protection against bollworms, but sucking pests remain a significant concern.

**Scouting and Economic Thresholds**

Regular scouting forms the foundation of any IPM programme. Inspect at least 20 plants per field every 7 to 10 days from crop emergence. Record pest counts and compare against economic thresholds. For jassids, apply control measures only when 2 or more nymphs are found per leaf. For whitefly, the threshold is 8 to 10 adults per leaf.

**Cultural Controls**

Deep summer ploughing exposes soil-dwelling pest pupae to heat and bird predation. Grow border crops of maize, sorghum, or sunflower to trap bollworms and facilitate their removal. Avoid ratoon cultivation which perpetuates pest and disease populations. Maintain proper field sanitation by destroying crop debris after harvest.

**Biological Controls**

Release Chrysoperla carnea (green lacewing) eggs at the rate of 50,000 per hectare at 30 and 60 days after sowing to control sucking pests. Trichogramma chilonis egg cards can be used for early bollworm management. Conservation of natural enemies like spiders, beetles, and parasitic wasps by avoiding broad-spectrum pesticides is essential.

**Pheromone Traps**

Install Helicoverpa armigera pheromone traps at the rate of 5 per hectare from 30 days after sowing. Monitor trap catches weekly. When catches exceed 8 to 10 moths per trap per week, prepare for larval emergence and plan spray timing accordingly.

**Chemical Control as a Last Resort**

When pest populations exceed economic thresholds despite cultural and biological measures, targeted chemical application may be necessary. Choose selective insecticides like Spinosad, Indoxacarb, or Emamectin benzoate that spare natural enemies. Rotate between chemical classes to delay resistance development.`,
  },
  {
    id: 7,
    title: "Organic Farming Transition: A Step-by-Step Guide",
    description: "Transitioning to organic farming can unlock premium markets and improve soil health. This guide covers the three-year transition process, NPOP certification, and marketing your produce.",
    category: "Crop Management",
    author: "Sunita Devi",
    date: "March 30, 2026",
    readTime: "10 min read",
    thumbnail: "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=900&q=80",
    content: `Transitioning from conventional to organic farming is a three-year commitment that transforms your soil ecosystem, production methods, and market linkages. The process requires patience, planning, and access to reliable information.

**Understanding the Transition Period**

Under the National Programme for Organic Production (NPOP), a field must be managed organically for at least 36 months before the crop can be certified as organic. During this period, you cannot use any synthetic fertilisers, pesticides, or genetically modified seeds. Your produce will be sold at conventional prices during this phase.

**Year One: Soil Building**

Focus on building soil organic matter. Incorporate green manures like dhaincha, sunhemp, or cowpea before land preparation. Apply well-composted farmyard manure at 10 to 15 tonnes per hectare. Introduce vermicomposting on your farm using cattle dung and crop residues. Begin producing biofertilisers like Rhizobium, Azotobacter, and PSB culture.

**Year Two: Pest and Nutrient Management**

By year two your soil biology begins to improve, making plants more resilient. Develop your on-farm inputs including neem seed kernel extract, Brahmastra, Panchagavya, and Jeevamrita. These preparations take time to learn but significantly reduce input costs compared to purchased organic inputs.

**Year Three: Certification Process**

Contact an NPOP-accredited certification body such as APOF, INDOCERT, or Lacon India. Submit your system plan, field history, and input records. An inspector will visit your farm for an initial assessment. Maintain meticulous records of all inputs, yields, and sales from the first day of transition.

**Premium Markets**

Certified organic produce commands 20 to 50 percent premiums depending on the commodity. Connect with organic aggregators, direct consumer groups, and export houses. Farmer Producer Organisations (FPOs) that achieve group certification can access export markets at lower per-farmer certification costs.`,
  },
  {
    id: 8,
    title: "Rainfall Deficiency and Drought Management Strategies",
    description: "Climate variability is increasing drought frequency across India. Learn water harvesting techniques, drought-tolerant variety selection, and crop insurance options to safeguard your livelihood.",
    category: "Weather",
    author: "Ramesh Kumar",
    date: "March 20, 2026",
    readTime: "8 min read",
    thumbnail: "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=900&q=80",
    content: `Drought is the most significant climate-related threat to Indian agriculture. With climate change intensifying variability, farmers need proactive strategies to manage both yield risk and livelihood risk when rainfall fails.

**Types of Drought**

Meteorological drought occurs when rainfall is significantly below normal for a defined period. Agricultural drought follows when soil moisture drops below the crop requirement threshold. Hydrological drought develops when surface and groundwater reserves are depleted. Understanding which type your area faces helps in selecting appropriate responses.

**Water Harvesting Structures**

Farm ponds, check dams, percolation tanks, and contour bunding can significantly increase water availability during dry spells. The National Watershed Management Programme provides subsidies for construction of these structures. A properly designed farm pond of 30 by 30 by 3 metres can store enough water for life-saving irrigation of one hectare of crops during a 15 to 20 day dry spell.

**Drought-Tolerant Crop Varieties**

State agriculture universities have released drought-tolerant varieties for most major crops. In maize, varieties like Vivek QPM 9 and HQPM 1 can withstand 10 to 15 day dry spells during grain filling. In paddy, Sahbhagi Dhan and DRR Dhan 42 are recommended for rainfed areas. Pearl millet and sorghum are inherently more drought-tolerant than maize or paddy and should be considered for high-risk areas.

**Crop Insurance Under PMFBY**

The Pradhan Mantri Fasal Bima Yojana covers losses due to drought, excess rainfall, floods, pests, and diseases. Enrol before the last date for your district, which is typically 5 to 7 days before sowing. Pay the applicable premium, which is capped at 1.5 percent for rabi cereals, 2 percent for kharif food and oilseed crops, and 5 percent for annual commercial and horticultural crops. Keep sowing records and report crop damage within 72 hours to your insurance company.

**Contingency Crop Planning**

If main season sowing is delayed beyond the optimal window due to delayed monsoon onset, switch to short-duration alternatives. Replace long-duration rice with medium-duration varieties. Replace main season groundnut with short-duration black gram. Your state agriculture department publishes contingency crop plans that are tailored to specific districts and rainfall scenarios.`,
  },
];

const CATEGORY_ICONS = {
  "Crop Management": <FaSeedling />,
  Weather: <FaCloudSun />,
  "Government Schemes": <FaLandmark />,
  Irrigation: <FaTint />,
  "Pest Management": <FaBug />,
};

export default function BlogDetail() {
  const { id } = useParams();
  const post = BLOG_POSTS.find((p) => p.id === parseInt(id, 10));
  const [isBookmarked, setIsBookmarked] = useState(false);

  useEffect(() => {
    setIsBookmarked(getBookmarks("articles").some((item) => item.id === parseInt(id, 10)));
  }, [id]);

  const handleToggleArticleBookmark = () => {
    if (!post) return;
    const updated = toggleBookmark("articles", post);
    setIsBookmarked(updated.some((item) => item.id === post.id));
  };

  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, [id]);

  if (!post) {
    return (
      <div className="blog-page blog-not-found">
        <FaLeaf className="empty-icon" />
        <h2>Article Not Found</h2>
        <p>The article you are looking for does not exist or may have been removed.</p>
        <Link to="/blog" id="blog-back-from-404" className="btn-back-blog">
          <FaArrowLeft /> Back to Blog
        </Link>
      </div>
    );
  }

  const paragraphs = post.content.split("\n\n");

  return (
    <div className="blog-page">
      <div className="blog-detail-wrap">
        <Link to="/blog" id="blog-back-nav" className="btn-back-blog">
          <FaArrowLeft /> Back to Blog
        </Link>

        <div className="blog-detail-hero">
          <img
            src={post.thumbnail}
            alt={post.title}
            className="blog-detail-thumb"
          />
          <div className="blog-detail-overlay" />
          <div className="blog-detail-hero-content">
            <div className="blog-card-category detail-cat">
              <span className="cat-icon">{CATEGORY_ICONS[post.category]}</span>
              {post.category}
            </div>
            <h1>{post.title}</h1>
            <button
              className={`detail-bookmark-btn ${isBookmarked ? "active" : ""}`}
              onClick={handleToggleArticleBookmark}
              aria-label={isBookmarked ? "Remove bookmark" : "Bookmark article"}
            >
              {isBookmarked ? <FaBookmark /> : <FaRegBookmark />} {isBookmarked ? "Saved" : "Bookmark"}
            </button>
            <div className="blog-detail-meta">
              <span className="meta-item">
                <FaUser /> {post.author}
              </span>
              <span className="meta-item">
                <FaCalendarAlt /> {post.date}
              </span>
              <span className="meta-item">
                <FaClock /> {post.readTime}
              </span>
            </div>
          </div>
        </div>

        <div className="blog-detail-content">
          {paragraphs.map((para, i) => {
            if (para.startsWith("**") && para.endsWith("**")) {
              return (
                <h3 key={i} className="blog-detail-h3">
                  {para.replace(/\*\*/g, "")}
                </h3>
              );
            }
            const parts = para.split(/(\*\*[^*]+\*\*)/g);
            return (
              <p key={i} className="blog-detail-para">
                {parts.map((part, j) =>
                  part.startsWith("**") && part.endsWith("**") ? (
                    <strong key={j}>{part.replace(/\*\*/g, "")}</strong>
                  ) : (
                    part
                  )
                )}
              </p>
            );
          })}
        </div>

        <div className="blog-detail-footer">
          <Link to="/blog" id="blog-back-bottom" className="btn-back-blog">
            <FaArrowLeft /> Back to All Articles
          </Link>
        </div>
      </div>
    </div>
  );
}