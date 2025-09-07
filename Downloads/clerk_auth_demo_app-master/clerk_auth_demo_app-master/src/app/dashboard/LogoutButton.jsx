'use client';

import { useAuth } from '@clerk/nextjs'; // Import Clerk hooks

const LogoutButton = ({ onClick }) => {
  return (
    <button
      className="bg-red-500 text-white px-4 py-2 rounded"
      onClick={onClick} // Attach the passed onClick function
    >
      Logout
    </button>
  );
};

export default LogoutButton;
