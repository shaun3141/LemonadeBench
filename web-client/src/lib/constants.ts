// Game constants - matching backend values

// Recipe: how many cups each ingredient makes
export const CUPS_PER_LEMON = 4;
export const CUPS_PER_SUGAR_BAG = 10;
export const CUPS_PER_ICE_BAG = 5;

// Base prices in cents
export const BASE_PRICES = {
  lemon: 25,
  sugar_bag: 100,
  cup: 5,
  ice_bag: 50,
} as const;

// Game configuration
export const TOTAL_GAME_DAYS = 14;
export const STARTING_CASH = 2000; // cents
export const OPTIMAL_PRICE = 50; // cents

// Pricing bounds (cents)
export const MIN_PRICE = 25;
export const MAX_PRICE = 200;
export const PRICE_STEP = 5;

// Advertising bounds (cents)
export const MIN_ADVERTISING = 0;
export const MAX_ADVERTISING = 500;
export const ADVERTISING_STEP = 25;

