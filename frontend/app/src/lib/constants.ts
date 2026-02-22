export const SWIPE_THRESHOLD = 100;
export const SWIPE_VELOCITY_THRESHOLD = 500;
export const QUEUE_PREFETCH_THRESHOLD = 3;
export const QUEUE_BATCH_SIZE = 5;
export const CARD_ROTATION_FACTOR = 0.1;

export const PROVIDER_LOGO_MAP: Record<string, string> = {
  Netflix: "/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg",
  "Amazon Video": "/emthp39XA2YScoYL1p0sdbAH2WA.jpg",
  "Amazon Prime Video": "/emthp39XA2YScoYL1p0sdbAH2WA.jpg",
  "Apple TV": "/6uhKBfmtzFqOcLousHwZuzcrScK.jpg",
  "Apple TV Store": "/6uhKBfmtzFqOcLousHwZuzcrScK.jpg",
  "Disney Plus": "/7rwgEs15tFwyR9NPQ5vpzxTj19Q.jpg",
  "Google Play Movies": "/tbEdFQDwx5LEVr8WpSeXQSIirVq.jpg",
  Hulu: "/gbyLHzl4eYP0ogPLlgvPq9FRpbK.jpg",
  "HBO Max": "/nmU4KykkOMw9D8kLKZDbKiLckcz.jpg",
  "Paramount Plus": "/xbhHHa1YgtpwhC8lb1NQ3ACVsLd.jpg",
  "Paramount Plus Essential": "/xbhHHa1YgtpwhC8lb1NQ3ACVsLd.jpg",
  YouTube: "/8VCV78prwd9QzZnEm0ReO6bERDa.jpg",
  "Fandango At Home": "/shq88b09fnSwCsZslh8v4pCRJFI.jpg",
  Vudu: "/shq88b09fnSwCsZslh8v4pCRJFI.jpg",
  Peacock: "/8GNy2MM89WgCUHBzGDVpnFXnaSK.jpg",
  Plex: "/9Dw85ssNbRB1xmsSSP2T7kYIZe.jpg",
  Max: "/nmU4KykkOMw9D8kLKZDbKiLckcz.jpg",
  Showtime: "/2PTMv3i1hwyJVGBGxXBf89iL00q.jpg",
  Crunchyroll: "/8I7SFkerpp0bpVcMFJhKGZYGQTN.jpg",
};

export const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/original";

export function getProviderLogoUrl(providerName: string): string {
  const logoPath = PROVIDER_LOGO_MAP[providerName];
  return logoPath
    ? `${TMDB_IMAGE_BASE}${logoPath}`
    : `https://via.placeholder.com/100x100/3b82f6/ffffff?text=${encodeURIComponent(providerName.slice(0, 2))}`;
}

