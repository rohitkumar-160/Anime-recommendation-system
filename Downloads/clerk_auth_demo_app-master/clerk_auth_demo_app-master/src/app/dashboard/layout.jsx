// DashboardLayout.jsx
import { UserButton } from '@clerk/nextjs';

const DashboardLayout = ({ children }) => {
  return (
    <div>
       
      <div>{children}</div>
    </div>
  );
};

export default DashboardLayout;
