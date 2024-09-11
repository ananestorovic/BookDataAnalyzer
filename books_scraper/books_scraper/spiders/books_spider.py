import scrapy
import re
from books_scraper.items import BooksScraperItem


class BooksSpider(scrapy.Spider):
    name = "books_spider"
    allowed_domains = ['www.knjizare-vulkan.rs']
    start_urls = ["https://www.knjizare-vulkan.rs/domace-knjige"]

    def parse(self, response):
        books = response.css(".item-data")

        for book in books:
            title = book.css(".product-link::attr(title)").get()
            details_link = book.css(".product-link::attr(href)").get()

            if details_link:
                yield response.follow(details_link, callback=self.parse_details)
            else:
                self.logger.info(f"Details link not found for the book titled: {title}")

        next_url = response.css("li.next>a::attr(href)").extract_first("")

        if next_url:
            page_number = re.search(r'\d+', next_url).group()
            next_page_url = f"{self.start_urls[0]}/page-{page_number}"
            self.logger.info(f"Next page URL: {next_page_url}")
            yield scrapy.Request(response.urljoin(next_page_url), callback=self.parse)

    def parse_details(self, response):
        self.logger.info(f"Parsing details page: {response.url}")

        title = response.css(".block .product-details-info .title h1 span::text").get(default='N/A')
        price = response.css(".product-price-without-discount-value::text").get() or response.css(
            ".product-price-value::text").get()
        price = float(price.split(',')[0].replace('.', '')) if price else 0.0

        author = response.xpath("//tr[td[contains(text(), 'Autor')]]/td[2]/a/text()").get(default='').strip()
        category = response.xpath("//tr[td[contains(text(), 'Kategorija')]]/td[2]/a/text()").get(default='').strip()
        publisher = response.xpath("//tr[td[contains(text(), 'Izdavač')]]/td[2]/a/text()").get(default='').strip()
        binding = response.xpath("//tr[td[contains(text(), 'Povez')]]/td[2]/text()").get(default='').strip()
        format = response.xpath("//tr[td[contains(text(), 'Format')]]/td[2]/text()").get(default='').strip()
        pages = response.xpath("//tr[td[contains(text(), 'Strana')]]/td[2]/text()").get(default='-1').strip()
        year = response.xpath("//tr[td[contains(text(), 'Godina')]]/td[2]/text()").get(default='').strip()

        description = ' '.join(response.css("#tab_product_description::text").getall()).strip()

        item = BooksScraperItem(
            code=response.css(".code span::text").get(default='N/A'),
            title=title,
            price=price,
            author=author,
            category=category,
            publisher=publisher,
            binding=binding,
            format=format,
            pages=int(pages) if pages.isdigit() else -1,
            year=year,
            description=description
        )
        yield item

