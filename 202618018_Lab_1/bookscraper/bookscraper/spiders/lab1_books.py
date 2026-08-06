import scrapy

class Lab1BooksSpider(scrapy.Spider):
    name = "lab1_books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/"]

    page_count = 1
    max_pages = 5

    def parse(self, response):
        book_links = response.css("article.product_pod h3 a::attr(href)").getall()

        for link in book_links:
            yield response.follow(link, callback=self.parse_book)

        next_page = response.css("li.next a::attr(href)").get()

        if next_page and self.page_count < self.max_pages:
            self.page_count += 1
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        title = response.css("div.product_main h1::text").get()
        price = response.css("p.price_color::text").get()
        rating = response.css("p.star-rating::attr(class)").get()
        availability = response.css("p.availability::text").getall()

        category = response.css("ul.breadcrumb li:nth-child(3) a::text").get()

        description = response.css("#product_description + p::text").get()

        upc = response.css("table.table.table-striped tr:nth-child(1) td::text").get()

        reviews = response.css("table.table.table-striped tr:nth-child(7) td::text").get()

        product_url = response.url

        yield {
            "Title": title,
            "Category": category,
            "Price": price,
            "Rating": rating.replace("star-rating ", ""),
            "Availability": " ".join(availability).strip(),
            "Description": description,
            "UPC": upc,
            "Number_of_Reviews": reviews,
            "Product_URL": product_url
}