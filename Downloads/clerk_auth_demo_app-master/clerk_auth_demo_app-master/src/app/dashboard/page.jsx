// AniWatchPortal.jsx
'use client';

import { useUser } from '@clerk/nextjs';
import { useState, useEffect } from 'react';
import { UserButton } from '@clerk/nextjs';
import { FaHome, FaStar, FaListAlt } from 'react-icons/fa'; 

const AniWatchPortal = () => {
  const { user } = useUser();
  const [animeData, setAnimeData] = useState([
    { title: "Attack on Titan", genre: "action", img: "/static/at.jpg" },
    { title: "One Piece", genre: "adventure", img: "/static/op.jpg" },
    { title: "Naruto", genre: "action", img: "/static/na.jpg" },
    { title: "Your Name", genre: "romance", img: "/static/yn.jpg" },
    { title: "Demon Slayer", genre: "action", img: "/static/ds.jpg" },
    { title: "My Hero Academia", genre: "action", img: "/static/ma.jpg" },
    { title: "Kaguya-sama", genre: "comedy", img: "/static/ks.jpg" },
    { title: "Re:Zero", genre: "adventure", img: "/static/rz.jpg" },
    { title: "Heroic Bloodshed", genre: "action", img: "/static/bc.jpg" },
    { title: "Spirited Away", genre: "adventure", img: "/static/sa.jpg" },
    { title: "Military Action", genre: "action", img: "/static/mat.jpg" },
    { title: "Castle in the Sky", genre: "romance", img: "/static/cs.jpg" },
    { title: "Espionage", genre: "action", img: "/static/es.jpg" },
    { title: "Wuxia Action", genre: "action", img: "/static/wa.jpg" },
    { title: "My Neighbor Totoro", genre: "comedy", img: "/static/mnt.jpg" },
    { title: "Princess Mononoke", genre: "adventure", img: "/static/pm.jpg" },
    { title: "Whisper of the Heart", genre: "romance", img: "/static/wh.jpg" },
    { title: "The Garden of Words", genre: "romance", img: "/static/gw.jpg" }
  ]);
  
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [sortOrder, setSortOrder] = useState('default');
  const itemsPerPage = 15;

  const filterAnime = () => {
    return animeData.filter(anime => {
      const titleMatch = anime.title.toLowerCase().includes(searchTerm.toLowerCase());
      const genreMatch = selectedGenre === 'all' || anime.genre === selectedGenre;
      return titleMatch && genreMatch;
    });
  };

  const sortAnime = (data) => {
    let sortedData = [...data];
    if (sortOrder === 'asc') {
      sortedData.sort((a, b) => a.title.localeCompare(b.title));
    } else if (sortOrder === 'desc') {
      sortedData.sort((a, b) => b.title.localeCompare(a.title));
    }
    return sortedData;
  };

  const paginate = (data) => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return data.slice(startIndex, startIndex + itemsPerPage);
  };

  const generatePagination = (data) => {
    const pageCount = Math.ceil(data.length / itemsPerPage);
    return Array.from({ length: pageCount }, (_, i) => i + 1);
  };

  const filteredData = filterAnime();
  const sortedData = sortAnime(filteredData);
  const paginatedData = paginate(sortedData);
  const pageCount = generatePagination(sortedData);

  useEffect(() => {
    window.scrollTo(0, 0); // Scroll to top on page load
  }, [currentPage]);

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', backgroundColor: '#101010', color: 'white', margin: 0, padding: 0 }}>
    <header style={{ backgroundColor: '#130f40', backgroundImage: 'linear-gradient(315deg, #130f40 0%, #000000 74%)', padding: '20px 0', textAlign: 'center', boxShadow: '0 8px 20px rgba(0, 0, 0, 0.3)', borderBottom: '2px solid rgba(255, 255, 255, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px' }}>
      <img src="/static/av.jpg" alt="AniVault Logo" style={{ maxWidth: '80px', height: 'auto', borderRadius: '50%' }} />
      <h1 style={{ margin: 0, fontSize: '3rem', color: '#FF5733', fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif', letterSpacing: '1px', textShadow: '2px 2px 5px rgba(0, 0, 0, 0.5)' }}>AniVault</h1>
    </header>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 20px' }}>
        <nav>
          <a href="http://localhost:5000/prof" style={{ color: '#f1f1f1', fontSize: '1rem', padding: '10px 20px', textDecoration: 'none', margin: '0 15px', borderRadius: '5px' }}> Home</a>
          <a href="http://localhost:5000/genres" style={{ color: '#f1f1f1', fontSize: '1rem', padding: '10px 20px', textDecoration: 'none', margin: '0 15px', borderRadius: '5px' }}> Recommendation</a>
          <a href="http://localhost:5000/animelist" style={{ color: '#f1f1f1', fontSize: '1rem', padding: '10px 20px', textDecoration: 'none', margin: '0 15px', borderRadius: '5px' }}> Anime List</a>
          
        </nav>
        <div>
        <input
            type="text"
            placeholder="Search for anime..."
            style={{ padding: '12px', width: '250px', borderRadius: '25px', border: 'none', backgroundColor: '#333', color: 'white', fontSize: '16px' }}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
         
         
          <select style={{ padding: '12px', backgroundColor: '#444', color: 'white', fontSize: '16px', borderRadius: '5px', border: 'none', margin: '0 10px' }} onChange={(e) => setSelectedGenre(e.target.value)}>
            <option value="all">All Genres</option>
            <option value="action">Action</option>
            <option value="adventure">Adventure</option>
            <option value="romance">Romance</option>
            <option value="comedy">Comedy</option>
          </select>
          <select style={{ padding: '12px', backgroundColor: '#444', color: 'white', fontSize: '16px', borderRadius: '5px', border: 'none', margin: '0 10px' }} onChange={(e) => setSortOrder(e.target.value)}>
            <option value="default">Sort by Title</option>
            <option value="asc">A - Z</option>
            <option value="desc">Z - A</option>
          </select>
           </div>
          <UserButton afterSignOutUrl="http://localhost:5000" />
       
      </div>

      <div style={{ position: 'relative', overflow: 'hidden', marginTop: '20px' }}>
        <div style={{ display: 'flex', animation: 'scroll 30s linear infinite' }}>
          <img src="/static/b15.jpg" alt="Banner 1" style={{ width: '100%', height: '200px', objectFit: 'cover' }} />
          <img src="/static/b12.jpg" alt="Banner 2" style={{ width: '100%', height: '200px', objectFit: 'cover' }} />
          <img src="/static/b13.jpg" alt="Banner 3" style={{ width: '100%', height: '200px', objectFit: 'cover' }} />
         
        </div>
      </div>

      <div className="anime-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px', padding: '20px', justifyItems: 'center' }}>
        {paginatedData.map((anime) => (
          <div className="anime-card" style={{ backgroundColor: '#2c2c2c', borderRadius: '10px', overflow: 'hidden', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)', width: '250px' }} key={anime.title}>
            <img src={anime.img} alt={anime.title} style={{ width: '100%', height: '300px', objectFit: 'cover' }} />
            <h3 style={{ padding: '10px', margin: 0, textAlign: 'center', backgroundColor: '#444', color: '#fff' }}>{anime.title}</h3>
          </div>
        ))}
      </div>

      <div className="pagination" style={{ textAlign: 'center', padding: '20px' }}>
        {pageCount.map((page) => (
          <button
            key={page}
            onClick={() => setCurrentPage(page)}
            style={{
              padding: '12px 20px',
              backgroundColor: '#444',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              margin: '0 5px',
              fontSize: '1rem',
              transition: 'background-color 0.3s',
            }}
          >
            {page}
          </button>
        ))}
      </div>
    </div>
  );
};

export default AniWatchPortal;
