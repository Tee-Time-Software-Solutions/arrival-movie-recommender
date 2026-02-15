import { config } from "./core/config"
import { useState, useEffect, useRef } from "react"

interface CastMember {
  name: string
  role_type: string
  profile_path: string
}

interface MovieProvider {
  name: string
  provider_type: string
}

interface Movie {
  movie_id: string
  tmdb_id: string
  title: string
  poster_url: string
  release_year: number
  rating: number
  genres: string[]
  is_adult: boolean
  synopsis: string
  cast: CastMember[]
  trailer_url: string
  runtime: number
  movie_providers: MovieProvider[]
}

function App() {
  const [movie, setMovie] = useState<Movie | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [showProviders, setShowProviders] = useState(false)
  const [providerFilter, setProviderFilter] = useState<string>("all")
  const cardRef = useRef<HTMLDivElement>(null)

  const fetchMovie = async () => {
    try {
      const response = await fetch(`${config.BASE_API_URL}movies/feed`)
      const data = await response.json()
      setMovie(data)
      setShowDetails(false)
      setShowProviders(false)
      setProviderFilter("all")
      setDragOffset({ x: 0, y: 0 })
    } catch (error) {
      console.error("Error fetching movie:", error)
    }
  }

  const getYoutubeVideoId = (url: string): string | null => {
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&]+)/)
    return match ? match[1] : null
  }

  const getFilteredProviders = () => {
    if (!movie) return []
    if (providerFilter === "all") return movie.movie_providers
    return movie.movie_providers.filter(p => p.provider_type === providerFilter)
  }

  const getProviderLogo = (providerName: string): string => {
    // TMDB provider logo mapping (using TMDB's image CDN)
    const logoMap: Record<string, string> = {
      "Netflix": "/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg",
      "Amazon Video": "/emthp39XA2YScoYL1p0sdbAH2WA.jpg",
      "Amazon Prime Video": "/emthp39XA2YScoYL1p0sdbAH2WA.jpg",
      "Apple TV": "/6uhKBfmtzFqOcLousHwZuzcrScK.jpg",
      "Apple TV Store": "/6uhKBfmtzFqOcLousHwZuzcrScK.jpg",
      "Disney Plus": "/7rwgEs15tFwyR9NPQ5vpzxTj19Q.jpg",
      "Google Play Movies": "/tbEdFQDwx5LEVr8WpSeXQSIirVq.jpg",
      "Hulu": "/gbyLHzl4eYP0ogPLlgvPq9FRpbK.jpg",
      "HBO Max": "/nmU4KykkOMw9D8kLKZDbKiLckcz.jpg",
      "Paramount Plus": "/xbhHHa1YgtpwhC8lb1NQ3ACVsLd.jpg",
      "Paramount Plus Essential": "/xbhHHa1YgtpwhC8lb1NQ3ACVsLd.jpg",
      "YouTube": "/8VCV78prwd9QzZnEm0ReO6bERDa.jpg",
      "Fandango At Home": "/shq88b09fnSwCsZslh8v4pCRJFI.jpg",
      "Vudu": "/shq88b09fnSwCsZslh8v4pCRJFI.jpg",
      "Peacock": "/8GNy2MM89WgCUHBzGDVpnFXnaSK.jpg",
      "Plex": "/9Dw85ssNbRB1xmsSSP2T7kYIZe.jpg",
      "JustWatch TV": "/6Dvju7pOHPyH7UlHkp5Xdz2ywUg.jpg",
      "Max": "/nmU4KykkOMw9D8kLKZDbKiLckcz.jpg",
      "Showtime": "/2PTMv3i1hwyJVGBGxXBf89iL00q.jpg",
      "Crunchyroll": "/8I7SFkerpp0bpVcMFJhKGZYGQTN.jpg",
    }

    const logoPath = logoMap[providerName]
    return logoPath
      ? `https://image.tmdb.org/t/p/original${logoPath}`
      : `https://via.placeholder.com/100x100/3b82f6/ffffff?text=${encodeURIComponent(providerName.slice(0, 2))}`
  }

  useEffect(() => {
    fetchMovie()
  }, [])

  const handleDragStart = (clientX: number, clientY: number) => {
    setIsDragging(true)
    setDragStart({ x: clientX, y: clientY })
  }

  const handleDragMove = (clientX: number, clientY: number) => {
    if (!isDragging) return
    const deltaX = clientX - dragStart.x
    const deltaY = clientY - dragStart.y
    setDragOffset({ x: deltaX, y: deltaY })
  }

  const handleDragEnd = () => {
    if (!isDragging) return
    setIsDragging(false)

    const threshold = 100

    // Swipe right - like (fetch next movie)
    if (dragOffset.x > threshold) {
      fetchMovie()
    }
    // Swipe left - dislike (n/a for now)
    else if (dragOffset.x < -threshold) {
      fetchMovie()
    }
    // Swipe down - show details
    else if (dragOffset.y > threshold) {
      setShowDetails(true)
      setDragOffset({ x: 0, y: 0 })
    }
    // Swipe up - haven't seen (n/a for now)
    else if (dragOffset.y < -threshold) {
      fetchMovie()
    }
    // Reset position if threshold not met
    else {
      setDragOffset({ x: 0, y: 0 })
    }
  }

  if (!movie) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <p className="text-white text-xl">Loading...</p>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="relative w-full max-w-md">
        <div
          ref={cardRef}
          className="relative cursor-grab active:cursor-grabbing"
          style={{
            transform: `translate(${dragOffset.x}px, ${dragOffset.y}px) rotate(${dragOffset.x * 0.1}deg)`,
            transition: isDragging ? "none" : "transform 0.3s ease",
          }}
          onMouseDown={(e) => handleDragStart(e.clientX, e.clientY)}
          onMouseMove={(e) => handleDragMove(e.clientX, e.clientY)}
          onMouseUp={handleDragEnd}
          onMouseLeave={handleDragEnd}
          onTouchStart={(e) => handleDragStart(e.touches[0].clientX, e.touches[0].clientY)}
          onTouchMove={(e) => handleDragMove(e.touches[0].clientX, e.touches[0].clientY)}
          onTouchEnd={handleDragEnd}
        >
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
            <img
              src={movie.poster_url}
              alt={movie.title}
              className="w-full h-[600px] object-cover select-none"
              draggable={false}
            />
            <div className="p-6">
              <h1 className="text-3xl font-bold mb-2">{movie.title}</h1>
              <div className="flex items-center gap-4 text-gray-600 mb-4">
                <span>{movie.release_year}</span>
                <span>⭐ {movie.rating.toFixed(1)}</span>
                <span>{movie.runtime} min</span>
              </div>
              <div className="flex flex-wrap gap-2 mb-4">
                {movie.genres.map((genre) => (
                  <span
                    key={genre}
                    className="px-3 py-1 bg-gray-200 rounded-full text-sm"
                  >
                    {genre}
                  </span>
                ))}
              </div>

              {showDetails && (
                <div className="mt-6 space-y-4 border-t pt-4">
                  <div>
                    <h3 className="font-semibold mb-2">Synopsis</h3>
                    <p className="text-gray-700 text-sm">{movie.synopsis}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-2">Trailer</h3>
                    {getYoutubeVideoId(movie.trailer_url) && (
                      <div className="relative w-full" style={{ paddingBottom: "56.25%" }}>
                        <iframe
                          className="absolute top-0 left-0 w-full h-full rounded-lg"
                          src={`https://www.youtube.com/embed/${getYoutubeVideoId(movie.trailer_url)}`}
                          title="YouTube video player"
                          frameBorder="0"
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                          allowFullScreen
                        />
                      </div>
                    )}
                  </div>

                  <div>
                    <h3 className="font-semibold mb-2">Cast</h3>
                    <div className="space-y-2">
                      {movie.cast.map((member) => (
                        <div key={member.name} className="flex items-center gap-3">
                          <img
                            src={member.profile_path}
                            alt={member.name}
                            className="w-12 h-12 rounded-full object-cover"
                          />
                          <div>
                            <p className="font-medium text-sm">{member.name}</p>
                            <p className="text-gray-600 text-xs">{member.role_type}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <button
                      onClick={() => setShowProviders(!showProviders)}
                      className="w-full flex items-center justify-between bg-gray-100 hover:bg-gray-200 px-4 py-3 rounded-lg font-semibold transition"
                    >
                      <span>Watch On ({movie.movie_providers.length} providers)</span>
                      <span>{showProviders ? "▲" : "▼"}</span>
                    </button>

                    {showProviders && (
                      <div className="mt-3 space-y-3">
                        <div className="flex gap-2">
                          <button
                            onClick={() => setProviderFilter("all")}
                            className={`px-3 py-1 rounded text-xs font-medium transition ${
                              providerFilter === "all"
                                ? "bg-blue-600 text-white"
                                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                            }`}
                          >
                            All ({movie.movie_providers.length})
                          </button>
                          <button
                            onClick={() => setProviderFilter("flatrate")}
                            className={`px-3 py-1 rounded text-xs font-medium transition ${
                              providerFilter === "flatrate"
                                ? "bg-green-600 text-white"
                                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                            }`}
                          >
                            Stream ({movie.movie_providers.filter(p => p.provider_type === "flatrate").length})
                          </button>
                          <button
                            onClick={() => setProviderFilter("rent")}
                            className={`px-3 py-1 rounded text-xs font-medium transition ${
                              providerFilter === "rent"
                                ? "bg-orange-600 text-white"
                                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                            }`}
                          >
                            Rent ({movie.movie_providers.filter(p => p.provider_type === "rent").length})
                          </button>
                          <button
                            onClick={() => setProviderFilter("buy")}
                            className={`px-3 py-1 rounded text-xs font-medium transition ${
                              providerFilter === "buy"
                                ? "bg-purple-600 text-white"
                                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                            }`}
                          >
                            Buy ({movie.movie_providers.filter(p => p.provider_type === "buy").length})
                          </button>
                        </div>

                        <div className="overflow-x-auto">
                          <div className="flex gap-4 pb-2">
                            {getFilteredProviders().map((provider, idx) => (
                              <div
                                key={idx}
                                className="flex flex-col items-center flex-shrink-0"
                              >
                                <div className={`w-16 h-16 rounded-xl overflow-hidden border-2 ${
                                  provider.provider_type === "flatrate"
                                    ? "border-green-500"
                                    : provider.provider_type === "rent"
                                    ? "border-orange-500"
                                    : "border-purple-500"
                                } bg-white shadow-md`}>
                                  <img
                                    src={getProviderLogo(provider.name)}
                                    alt={provider.name}
                                    className="w-full h-full object-cover"
                                  />
                                </div>
                                <span className="text-[10px] mt-1 text-center max-w-[80px] text-gray-700 leading-tight">
                                  {provider.name}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6 text-center text-white text-sm">
          <p>← Swipe Left: Dislike | Swipe Right: Like →</p>
          <p>↑ Swipe Up: Haven't Seen | Swipe Down: Details ↓</p>
        </div>
      </div>
    </div>
  )
}

export default App
