SELECT ss_customer_sk, SUM(ss_ext_sales_price) AS total_sales
FROM store_sales
GROUP BY ss_customer_sk
ORDER BY total_sales DESC
LIMIT 10;
