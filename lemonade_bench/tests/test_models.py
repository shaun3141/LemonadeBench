# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Tests for the data models in models.py.

Covers:
- Enums (Weather, CustomerMood, StandUpgrade)
- Bulk pricing calculations
- LemonadeAction dataclass
- LemonadeObservation dataclass
- GameConfig
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    Weather,
    CustomerMood,
    StandUpgrade,
    UpgradeInfo,
    BulkTier,
    BulkPricing,
    UPGRADE_CATALOG,
    BULK_PRICING,
    calculate_bulk_cost,
    LemonadeAction,
    LemonadeObservation,
    GameConfig,
    MarketHints,
    Location,
    LocationInfo,
    LOCATION_CATALOG,
)


class TestWeatherEnum:
    """Tests for the Weather enum."""

    def test_weather_values(self):
        """Test all weather values are correct."""
        assert Weather.SUNNY.value == "sunny"
        assert Weather.CLOUDY.value == "cloudy"
        assert Weather.RAINY.value == "rainy"
        assert Weather.HOT.value == "hot"
        assert Weather.STORMY.value == "stormy"

    def test_weather_count(self):
        """Test we have exactly 5 weather types."""
        assert len(Weather) == 5

    def test_weather_from_string(self):
        """Test creating Weather from string value."""
        assert Weather("sunny") == Weather.SUNNY
        assert Weather("hot") == Weather.HOT

    def test_weather_invalid_value_raises(self):
        """Test invalid weather value raises ValueError."""
        with pytest.raises(ValueError):
            Weather("tornado")


class TestCustomerMoodEnum:
    """Tests for the CustomerMood enum."""

    def test_customer_mood_values(self):
        """Test all customer mood values."""
        assert CustomerMood.HAPPY.value == "happy"
        assert CustomerMood.NEUTRAL.value == "neutral"
        assert CustomerMood.GRUMPY.value == "grumpy"

    def test_customer_mood_count(self):
        """Test we have exactly 3 mood types."""
        assert len(CustomerMood) == 3


class TestStandUpgradeEnum:
    """Tests for the StandUpgrade enum."""

    def test_cooler_upgrade_exists(self):
        """Test cooler upgrade value."""
        assert StandUpgrade.COOLER.value == "cooler"

    def test_upgrade_in_catalog(self):
        """Test all upgrades have catalog entries."""
        for upgrade in StandUpgrade:
            assert upgrade in UPGRADE_CATALOG
            info = UPGRADE_CATALOG[upgrade]
            assert isinstance(info, UpgradeInfo)
            assert info.id == upgrade
            assert info.cost > 0


class TestUpgradeCatalog:
    """Tests for the UPGRADE_CATALOG."""

    def test_cooler_upgrade_info(self):
        """Test cooler upgrade details."""
        cooler = UPGRADE_CATALOG[StandUpgrade.COOLER]
        assert cooler.name == "Ice Cooler"
        assert cooler.cost == 250  # $2.50
        assert "ice" in cooler.description.lower()
        assert "50%" in cooler.effect

    def test_all_upgrades_have_required_fields(self):
        """Test all upgrades have all required fields populated."""
        for upgrade, info in UPGRADE_CATALOG.items():
            assert info.id is not None
            assert info.name
            assert info.description
            assert info.cost > 0
            assert info.effect


class TestBulkPricing:
    """Tests for bulk pricing configuration."""

    def test_bulk_pricing_supplies_exist(self):
        """Test all supply types have bulk pricing."""
        assert "lemons" in BULK_PRICING
        assert "sugar" in BULK_PRICING
        assert "cups" in BULK_PRICING
        assert "ice" in BULK_PRICING

    def test_lemons_bulk_pricing(self):
        """Test lemon bulk pricing tiers."""
        lemons = BULK_PRICING["lemons"]
        assert lemons.unit_name == "Lemon"
        assert lemons.base_price == 25  # $0.25

        # Check tiers
        assert len(lemons.tiers) == 3
        single = lemons.tiers[0]
        assert single.name == "Single"
        assert single.quantity == 1
        assert single.discount_percent == 0.0

        dozen = lemons.tiers[1]
        assert dozen.name == "Dozen"
        assert dozen.quantity == 12
        assert dozen.discount_percent == 0.10

        crate = lemons.tiers[2]
        assert crate.name == "Crate"
        assert crate.quantity == 144
        assert crate.discount_percent == 0.20

    def test_sugar_bulk_pricing(self):
        """Test sugar bulk pricing tiers."""
        sugar = BULK_PRICING["sugar"]
        assert sugar.unit_name == "Bag"
        assert sugar.base_price == 50  # $0.50

    def test_cups_bulk_pricing(self):
        """Test cups bulk pricing tiers."""
        cups = BULK_PRICING["cups"]
        assert cups.unit_name == "Cup"
        assert cups.base_price == 5  # $0.05 per cup

    def test_ice_bulk_pricing(self):
        """Test ice bulk pricing tiers."""
        ice = BULK_PRICING["ice"]
        assert ice.unit_name == "Bag"
        assert ice.base_price == 25  # $0.25


class TestCalculateBulkCost:
    """Tests for the calculate_bulk_cost function."""

    def test_zero_quantity_returns_zero(self):
        """Test buying 0 items costs nothing."""
        assert calculate_bulk_cost("lemons", 0) == 0
        assert calculate_bulk_cost("sugar", 0) == 0

    def test_negative_quantity_returns_zero(self):
        """Test negative quantities return 0."""
        assert calculate_bulk_cost("lemons", -5) == 0

    def test_invalid_supply_type_returns_zero(self):
        """Test invalid supply types return 0."""
        assert calculate_bulk_cost("oranges", 10) == 0
        assert calculate_bulk_cost("", 10) == 0

    def test_single_lemon_full_price(self):
        """Test buying 1 lemon costs full price."""
        # 1 lemon at $0.25
        assert calculate_bulk_cost("lemons", 1) == 25

    def test_eleven_lemons_full_price(self):
        """Test buying 11 lemons (below dozen) costs full price."""
        # 11 lemons at $0.25 each
        assert calculate_bulk_cost("lemons", 11) == 11 * 25

    def test_dozen_lemons_discount(self):
        """Test buying 12 lemons gets 10% discount."""
        # 12 lemons * $0.25 = $3.00, with 10% off = $2.70
        base = 12 * 25
        discount = int(base * 0.10)
        assert calculate_bulk_cost("lemons", 12) == base - discount

    def test_fifty_lemons_dozen_discount(self):
        """Test buying 50 lemons gets dozen-tier discount (not yet crate)."""
        # 50 lemons * $0.25 = $12.50, with 10% off = $11.25
        base = 50 * 25
        discount = int(base * 0.10)
        assert calculate_bulk_cost("lemons", 50) == base - discount

    def test_crate_lemons_discount(self):
        """Test buying 144 lemons gets 20% crate discount."""
        # 144 lemons * $0.25 = $36.00, with 20% off = $28.80
        base = 144 * 25
        discount = int(base * 0.20)
        assert calculate_bulk_cost("lemons", 144) == base - discount

    def test_large_quantity_gets_best_discount(self):
        """Test buying 200 lemons gets crate discount."""
        # 200 lemons * $0.25 = $50.00, with 20% off = $40.00
        base = 200 * 25
        discount = int(base * 0.20)
        assert calculate_bulk_cost("lemons", 200) == base - discount

    def test_sugar_bulk_discount(self):
        """Test sugar bulk purchasing."""
        # 1 bag at $0.50
        assert calculate_bulk_cost("sugar", 1) == 50
        # 5 bags (case) at $0.50 * 5 = $2.50, with 10% off = $2.25
        assert calculate_bulk_cost("sugar", 5) == int(250 * 0.90)
        # 20 bags (pallet) at $0.50 * 20 = $10.00, with 20% off = $8.00
        assert calculate_bulk_cost("sugar", 20) == int(1000 * 0.80)

    def test_ice_bulk_discount(self):
        """Test ice bulk purchasing."""
        # 1 bag at $0.25
        assert calculate_bulk_cost("ice", 1) == 25
        # 5 bags (cooler) at $0.25 * 5 = $1.25, with 10% off = $1.13
        # Discount is int(125 * 0.10) = 12 cents, so 125 - 12 = 113 cents
        assert calculate_bulk_cost("ice", 5) == 113

    def test_cups_bulk_discount(self):
        """Test cups bulk purchasing."""
        # Cups are $0.05 each
        # 10 cups = $0.50 (Pack tier, no discount)
        assert calculate_bulk_cost("cups", 10) == 50
        # 50 cups = $2.50 base, with 10% off = $2.25 (Sleeve tier)
        assert calculate_bulk_cost("cups", 50) == int(250 * 0.90)
        # 250 cups = $12.50 base, with 20% off = $10.00 (Case tier)
        assert calculate_bulk_cost("cups", 250) == int(1250 * 0.80)


class TestLemonadeAction:
    """Tests for the LemonadeAction dataclass."""

    def test_minimal_action(self):
        """Test action with only required fields."""
        action = LemonadeAction(price_per_cup=50)
        assert action.price_per_cup == 50
        assert action.lemons_tier == 1
        assert action.lemons_count == 0
        assert action.sugar_tier == 1
        assert action.sugar_count == 0
        assert action.cups_tier == 1
        assert action.cups_count == 0
        assert action.ice_tier == 1
        assert action.ice_count == 0
        assert action.advertising_spend == 0
        assert action.buy_upgrade is None

    def test_full_action(self):
        """Test action with all fields populated."""
        action = LemonadeAction(
            price_per_cup=100,
            lemons_tier=2, lemons_count=2,   # 2 dozen = 24 lemons
            sugar_tier=2, sugar_count=1,     # 1 case = 5 bags
            cups_tier=3, cups_count=1,       # 1 case = 250 cups
            ice_tier=2, ice_count=2,         # 2 cooler packs = 10 bags
            advertising_spend=200,
            buy_upgrade="cooler",
        )
        assert action.price_per_cup == 100
        assert action.lemons_tier == 2
        assert action.lemons_count == 2
        assert action.sugar_tier == 2
        assert action.sugar_count == 1
        assert action.cups_tier == 3
        assert action.cups_count == 1
        assert action.ice_tier == 2
        assert action.ice_count == 2
        assert action.advertising_spend == 200
        assert action.buy_upgrade == "cooler"

    def test_action_with_zero_price(self):
        """Test action with zero price (free lemonade!)."""
        action = LemonadeAction(price_per_cup=0)
        assert action.price_per_cup == 0

    def test_action_with_high_price(self):
        """Test action with very high price."""
        action = LemonadeAction(price_per_cup=1000)
        assert action.price_per_cup == 1000


class TestLemonadeObservation:
    """Tests for the LemonadeObservation dataclass."""

    def test_observation_creation(self):
        """Test creating a basic observation."""
        obs = LemonadeObservation(
            day=1,
            weather="sunny",
            temperature=80,
            weather_forecast="cloudy",
            cash=2000,
            daily_revenue=0,
            daily_costs=0,
            daily_profit=0,
            cups_sold=0,
            cups_wasted=0,
            customers_served=0,
            customers_turned_away=0,
            lemons=10,
            sugar_bags=5.0,
            cups_available=50,
            customer_satisfaction=0.5,
            reputation=0.5,
            days_remaining=13,
            total_profit=0,
            done=False,
            reward=0.0,
        )
        assert obs.day == 1
        assert obs.weather == "sunny"
        assert obs.cash == 2000
        assert obs.done is False

    def test_observation_with_ice(self):
        """Test observation includes ice fields."""
        obs = LemonadeObservation(
            day=1,
            weather="hot",
            temperature=95,
            weather_forecast="sunny",
            cash=2000,
            daily_revenue=500,
            daily_costs=100,
            daily_profit=400,
            cups_sold=10,
            cups_wasted=0,
            customers_served=10,
            customers_turned_away=5,
            lemons=10,
            sugar_bags=5.0,
            cups_available=50,
            ice_bags=5,
            customer_satisfaction=0.8,
            reputation=0.6,
            days_remaining=10,
            total_profit=400,
            done=False,
            reward=4.0,
        )
        assert obs.ice_bags == 5

    def test_observation_with_spoilage(self):
        """Test observation includes spoilage tracking."""
        obs = LemonadeObservation(
            day=5,
            weather="cloudy",
            temperature=70,
            weather_forecast="rainy",
            cash=3000,
            daily_revenue=200,
            daily_costs=50,
            daily_profit=150,
            cups_sold=4,
            cups_wasted=1,
            customers_served=4,
            customers_turned_away=0,
            lemons=15,
            sugar_bags=4.0,
            cups_available=45,
            lemons_expiring_tomorrow=5,
            ice_expiring_tomorrow=3,
            lemons_spoiled=2,
            ice_melted=5,
            customer_satisfaction=0.7,
            reputation=0.65,
            days_remaining=9,
            total_profit=1500,
            done=False,
            reward=1.5,
        )
        assert obs.lemons_expiring_tomorrow == 5
        assert obs.lemons_spoiled == 2
        assert obs.ice_melted == 5

    def test_observation_with_upgrades(self):
        """Test observation includes upgrade tracking."""
        obs = LemonadeObservation(
            day=3,
            weather="sunny",
            temperature=85,
            weather_forecast="hot",
            cash=2500,
            daily_revenue=600,
            daily_costs=200,
            daily_profit=400,
            cups_sold=12,
            cups_wasted=0,
            customers_served=12,
            customers_turned_away=0,
            lemons=20,
            sugar_bags=8.0,
            cups_available=80,
            customer_satisfaction=0.9,
            reputation=0.7,
            days_remaining=11,
            total_profit=1200,
            owned_upgrades=["cooler"],
            upgrade_catalog=[
                {
                    "id": "cooler",
                    "name": "Ice Cooler",
                    "cost": 500,
                    "owned": True,
                }
            ],
            done=False,
            reward=4.0,
        )
        assert "cooler" in obs.owned_upgrades
        assert obs.upgrade_catalog is not None


class TestGameConfig:
    """Tests for the GameConfig dataclass."""

    def test_default_config(self):
        """Test default game configuration values."""
        config = GameConfig()
        assert config.total_days == 14
        assert config.starting_cash == 2000  # $20.00
        assert config.starting_lemons == 10
        assert config.starting_sugar == 5
        assert config.starting_cups == 50
        assert config.starting_ice == 5

    def test_default_supply_costs(self):
        """Test default supply costs."""
        config = GameConfig()
        assert config.lemon_cost == 25
        assert config.sugar_cost == 50  # $0.50 per bag
        assert config.cup_cost == 5
        assert config.ice_cost == 25  # $0.25 per bag

    def test_default_recipe(self):
        """Test default recipe per cup."""
        config = GameConfig()
        assert config.lemons_per_cup == 0.25  # 4 cups per lemon
        assert config.sugar_per_cup == 0.1  # 10 cups per bag
        assert config.ice_per_cup == 0.2  # 5 cups per bag

    def test_default_expiration(self):
        """Test default expiration settings."""
        config = GameConfig()
        assert config.lemon_shelf_life == 3
        assert config.ice_shelf_life == 1

    def test_custom_config(self):
        """Test custom game configuration."""
        config = GameConfig(
            total_days=7,
            starting_cash=5000,
            base_customers=100,
        )
        assert config.total_days == 7
        assert config.starting_cash == 5000
        assert config.base_customers == 100

    def test_config_price_settings(self):
        """Test price-related configuration."""
        config = GameConfig()
        assert config.price_sensitivity == 0.02
        assert config.max_price_tolerance == 200  # $2.00
        assert config.optimal_price == 50  # $0.50

    def test_config_ice_bonus(self):
        """Test ice demand bonus configuration."""
        config = GameConfig()
        assert config.ice_demand_bonus == 0.2  # 20%


class TestMarketHints:
    """Tests for the MarketHints dataclass."""

    def test_market_hints_creation(self):
        """Test creating market hints with two-stage model."""
        hints = MarketHints(
            # Foot traffic (people who stop by)
            foot_traffic_low=40,
            foot_traffic_high=50,
            weather_traffic_multiplier=1.3,
            # Conversion rates
            conversion_curve={25: 0.95, 50: 0.95, 75: 0.80, 100: 0.65},
            ice_conversion_bonus=0.2,
            # Price guidance
            optimal_price=50,
            price_demand_curve={25: 60, 50: 40, 100: 20},
            revenue_curve={25: 1500, 50: 2000, 100: 2000},  # price * customers
            optimal_revenue_price=75,  # Higher prices can maximize revenue
            max_cups_producible=100,
            limiting_resource="lemons",
            ingredient_cost_per_cup=20,
            break_even_price=20,
            suggested_production=40,
        )
        # Test foot traffic fields
        assert hints.foot_traffic_low == 40
        assert hints.foot_traffic_high == 50
        assert hints.weather_traffic_multiplier == 1.3
        # Test conversion fields
        assert hints.conversion_curve[50] == 0.95
        assert hints.ice_conversion_bonus == 0.2
        # Test price guidance
        assert hints.optimal_price == 50
        assert hints.optimal_revenue_price == 75
        assert hints.limiting_resource == "lemons"


class TestLocationEnum:
    """Tests for the Location enum."""

    def test_location_values(self):
        """Test all location values are correct."""
        assert Location.PARK.value == "park"
        assert Location.DOWNTOWN.value == "downtown"
        assert Location.MALL.value == "mall"
        assert Location.POOL.value == "pool"

    def test_location_count(self):
        """Test we have exactly 4 location types."""
        assert len(Location) == 4

    def test_location_from_string(self):
        """Test creating Location from string value."""
        assert Location("park") == Location.PARK
        assert Location("mall") == Location.MALL

    def test_location_invalid_value_raises(self):
        """Test invalid location value raises ValueError."""
        with pytest.raises(ValueError):
            Location("beach")


class TestLocationCatalog:
    """Tests for the LOCATION_CATALOG."""

    def test_all_locations_in_catalog(self):
        """Test all locations have catalog entries."""
        for location in Location:
            assert location in LOCATION_CATALOG
            info = LOCATION_CATALOG[location]
            assert isinstance(info, LocationInfo)
            assert info.id == location

    def test_park_location_info(self):
        """Test park location details (high volume play)."""
        park = LOCATION_CATALOG[Location.PARK]
        assert park.name == "Neighborhood Park"
        assert park.foot_traffic_multiplier == 1.2  # High volume location
        assert park.price_sensitivity == 0.018  # Moderate sensitivity
        assert park.weather_exposure == 1.0  # Full weather effect
        assert park.permit_cost == 0  # Free - home base location

    def test_downtown_location_info(self):
        """Test downtown location details (premium balanced)."""
        downtown = LOCATION_CATALOG[Location.DOWNTOWN]
        assert downtown.name == "Downtown"
        assert downtown.foot_traffic_multiplier == 1.0  # Standard traffic
        assert downtown.price_sensitivity == 0.012  # Tolerates higher prices
        assert downtown.weather_exposure == 0.7  # Good shelter from buildings

    def test_mall_location_info(self):
        """Test mall location details (premium niche, low volume)."""
        mall = LOCATION_CATALOG[Location.MALL]
        assert mall.name == "Shopping Mall"
        assert mall.foot_traffic_multiplier == 0.7  # LOW volume - trade traffic for price tolerance
        assert mall.price_sensitivity == 0.008  # Best price tolerance
        assert mall.weather_exposure == 0.0  # Indoor - no weather effect

    def test_pool_location_info(self):
        """Test pool location details (weather betting, hot day goldmine)."""
        pool = LOCATION_CATALOG[Location.POOL]
        assert pool.name == "Community Pool"
        assert pool.foot_traffic_multiplier == 0.9  # Slightly below average base
        assert pool.price_sensitivity == 0.020  # Budget-conscious normally
        assert pool.hot_weather_price_sensitivity == 0.010  # Premium prices on hot days!
        assert pool.weather_exposure == 1.8  # Highly amplified - best on hot days, worst on bad days

    def test_all_locations_have_required_fields(self):
        """Test all locations have all required fields populated."""
        for location, info in LOCATION_CATALOG.items():
            assert info.id is not None
            assert info.name
            assert info.description
            assert info.foot_traffic_multiplier > 0
            assert info.price_sensitivity > 0
            assert info.weather_exposure >= 0
            assert info.permit_cost >= 0  # Park is free (home base)


class TestLemonadeActionWithLocation:
    """Tests for LemonadeAction with location field."""

    def test_action_with_location(self):
        """Test action with location field."""
        action = LemonadeAction(
            price_per_cup=100,
            location="downtown",
        )
        assert action.price_per_cup == 100
        assert action.location == "downtown"

    def test_action_without_location(self):
        """Test action without location (stays at current)."""
        action = LemonadeAction(price_per_cup=50)
        assert action.location is None


class TestLemonadeObservationWithLocation:
    """Tests for LemonadeObservation with location fields."""

    def test_observation_with_location(self):
        """Test observation includes location fields."""
        obs = LemonadeObservation(
            day=1,
            weather="sunny",
            temperature=80,
            weather_forecast="cloudy",
            cash=2000,
            daily_revenue=0,
            daily_costs=0,
            daily_profit=0,
            cups_sold=0,
            cups_wasted=0,
            customers_served=0,
            customers_turned_away=0,
            lemons=10,
            sugar_bags=5.0,
            cups_available=50,
            customer_satisfaction=0.5,
            reputation=0.5,
            days_remaining=13,
            total_profit=0,
            current_location="downtown",
            location_catalog=[
                {"id": "park", "name": "Park", "is_current": False},
                {"id": "downtown", "name": "Downtown", "is_current": True},
            ],
            done=False,
            reward=0.0,
        )
        assert obs.current_location == "downtown"
        assert obs.location_catalog is not None
        assert len(obs.location_catalog) == 2


class TestGameConfigWithLocation:
    """Tests for GameConfig with location settings."""

    def test_default_starting_location(self):
        """Test default starting location is park."""
        config = GameConfig()
        assert config.starting_location == "park"

    def test_default_permit_cost(self):
        """Test default location permit cost."""
        config = GameConfig()
        assert config.location_permit_cost == 1000  # $10.00

    def test_custom_starting_location(self):
        """Test custom starting location."""
        config = GameConfig(starting_location="mall")
        assert config.starting_location == "mall"
