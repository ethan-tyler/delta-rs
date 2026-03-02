SELECT AVG(ss_quantity) AS avg_quantity, SUM(ss_ext_sales_price) AS total_sales
FROM store_sales
WHERE ss_sold_date_sk IS NOT NULL;
