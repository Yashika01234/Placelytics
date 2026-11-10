-- TalentPulse analytical SQL (SQLite >= 3.25). All data is SYNTHETIC.

-- name: branch_year_placement
-- CTEs + LEFT JOIN + LAG (year-over-year) + RANK (branch rank within year)
WITH pool AS (
  SELECT student_id, branch, batch_year FROM students WHERE eligible = 1 AND seeking_placement = 1),
placed AS (
  SELECT student_id, MAX(offer_lpa) AS best_offer FROM applications WHERE selected = 1 GROUP BY student_id),
by_branch_year AS (
  SELECT p.branch, p.batch_year, COUNT(*) AS pool_size, COUNT(pl.student_id) AS placed,
         ROUND(100.0 * COUNT(pl.student_id) / COUNT(*), 1) AS placement_pct,
         ROUND(AVG(pl.best_offer), 2) AS avg_lpa
  FROM pool p LEFT JOIN placed pl ON pl.student_id = p.student_id
  GROUP BY p.branch, p.batch_year)
SELECT branch, batch_year, pool_size, placed, placement_pct, avg_lpa,
       ROUND(placement_pct - LAG(placement_pct) OVER (PARTITION BY branch ORDER BY batch_year), 1) AS yoy_pp,
       RANK() OVER (PARTITION BY batch_year ORDER BY placement_pct DESC) AS rank_in_year
FROM by_branch_year
ORDER BY branch, batch_year;

-- name: funnel_by_branch
-- Student-level funnel eligible -> applied -> shortlisted -> interviewed -> selected
WITH per_student AS (
  SELECT s.student_id, s.branch, s.eligible,
         CASE WHEN COUNT(a.app_id) > 0 THEN 1 ELSE 0 END AS applied,
         COALESCE(MAX(a.shortlisted), 0) AS shortlisted,
         COALESCE(MAX(a.interviewed), 0) AS interviewed,
         COALESCE(MAX(a.selected), 0) AS selected
  FROM students s LEFT JOIN applications a ON a.student_id = s.student_id
  GROUP BY s.student_id, s.branch, s.eligible)
SELECT branch, SUM(eligible) AS eligible, SUM(applied) AS applied, SUM(shortlisted) AS shortlisted,
       SUM(interviewed) AS interviewed, SUM(selected) AS selected,
       ROUND(100.0 * SUM(selected) / SUM(eligible), 1) AS eligible_to_selected_pct
FROM per_student GROUP BY branch ORDER BY eligible_to_selected_pct DESC;

-- name: stage_conversion_by_year
-- Application-level stage conversion with LAG to expose interview-stage drift
WITH y AS (
  SELECT d.batch_year, COUNT(*) AS applied, SUM(a.shortlisted) AS sl, SUM(a.interviewed) AS iv, SUM(a.selected) AS sel
  FROM applications a JOIN drives d ON d.drive_id = a.drive_id GROUP BY d.batch_year)
SELECT batch_year, applied,
       ROUND(100.0 * sl / applied, 1) AS apply_to_shortlist_pct,
       ROUND(100.0 * iv / sl, 1) AS shortlist_to_interview_pct,
       ROUND(100.0 * sel / iv, 1) AS interview_to_offer_pct,
       ROUND(100.0 * iv / sl - LAG(100.0 * iv / sl) OVER (ORDER BY batch_year), 1) AS interview_conv_change_pp
FROM y ORDER BY batch_year;

-- name: salary_by_industry
-- Median / p90 without PERCENTILE_CONT (not in SQLite) using ROW_NUMBER + COUNT window functions
WITH offers AS (
  SELECT c.industry, a.offer_lpa,
         ROW_NUMBER() OVER (PARTITION BY c.industry ORDER BY a.offer_lpa) AS rn,
         COUNT(*) OVER (PARTITION BY c.industry) AS n
  FROM applications a JOIN drives d ON d.drive_id = a.drive_id JOIN companies c ON c.company_id = d.company_id
  WHERE a.selected = 1)
SELECT industry, COUNT(*) AS offers, ROUND(AVG(offer_lpa), 2) AS mean_lpa,
       ROUND(AVG(CASE WHEN rn IN ((n + 1) / 2, (n + 2) / 2) THEN offer_lpa END), 2) AS median_lpa,
       ROUND(MIN(CASE WHEN rn >= 0.9 * n THEN offer_lpa END), 2) AS p90_lpa,
       ROUND(MAX(offer_lpa), 2) AS max_lpa
FROM offers GROUP BY industry ORDER BY median_lpa DESC;

-- name: recruiter_trend
-- Recruiter offers per year with LAG (change), running total and share-of-year windows
WITH co AS (
  SELECT c.company_name, c.industry, d.batch_year, COUNT(*) AS offers, ROUND(AVG(a.offer_lpa), 2) AS avg_lpa
  FROM applications a JOIN drives d ON d.drive_id = a.drive_id JOIN companies c ON c.company_id = d.company_id
  WHERE a.selected = 1 GROUP BY c.company_id, c.company_name, c.industry, d.batch_year)
SELECT company_name, industry, batch_year, offers, avg_lpa,
       offers - LAG(offers) OVER (PARTITION BY company_name ORDER BY batch_year) AS offers_change,
       SUM(offers) OVER (PARTITION BY company_name ORDER BY batch_year) AS cumulative_offers,
       ROUND(100.0 * offers / SUM(offers) OVER (PARTITION BY batch_year), 1) AS share_of_year_pct
FROM co ORDER BY company_name, batch_year;

-- name: internship_impact
-- Placement rate by internship count within CGPA bands (controls for the obvious confounder)
WITH pool AS (
  SELECT s.student_id, MIN(s.internships, 3) AS internship_bucket,
         CASE WHEN s.cgpa < 7 THEN '1) <7.0' WHEN s.cgpa < 8 THEN '2) 7.0-7.99' ELSE '3) 8.0+' END AS cgpa_band,
         CASE WHEN EXISTS (SELECT 1 FROM applications a WHERE a.student_id = s.student_id AND a.selected = 1) THEN 1 ELSE 0 END AS placed
  FROM students s WHERE s.eligible = 1 AND s.seeking_placement = 1)
SELECT cgpa_band, internship_bucket, COUNT(*) AS pool_size, SUM(placed) AS placed,
       ROUND(100.0 * SUM(placed) / COUNT(*), 1) AS placement_pct
FROM pool GROUP BY cgpa_band, internship_bucket ORDER BY cgpa_band, internship_bucket;

-- name: skill_gap
-- Opening-weighted recruiter demand vs share of eligible students holding the skill
WITH dem AS (
  SELECT ds.skill, 1.0 * SUM(d.openings) / (SELECT SUM(openings) FROM drives) AS demand_share
  FROM drive_skills ds JOIN drives d ON d.drive_id = ds.drive_id GROUP BY ds.skill),
sup AS (
  SELECT ss.skill, 1.0 * COUNT(*) / (SELECT COUNT(*) FROM students WHERE eligible = 1) AS supply_share
  FROM student_skills ss JOIN students s ON s.student_id = ss.student_id WHERE s.eligible = 1 GROUP BY ss.skill)
SELECT dem.skill, ROUND(100 * demand_share, 1) AS demand_pct, ROUND(100 * COALESCE(supply_share, 0), 1) AS supply_pct,
       ROUND(100 * (demand_share - COALESCE(supply_share, 0)), 1) AS gap_pp,
       RANK() OVER (ORDER BY demand_share - COALESCE(supply_share, 0) DESC) AS gap_rank
FROM dem LEFT JOIN sup ON sup.skill = dem.skill ORDER BY gap_rank;
